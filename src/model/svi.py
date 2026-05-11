"""
SVI (Stochastic Volatility Inspired) raw parametrization de Gatheral.

w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

donde
    k = log(K/F)                        log-moneyness
    w(k) = sigma_imp(k)^2 * T            total implied variance
    params = (a, b, rho, m, sigma)

Derivadas analiticas (utiles luego para extraer densidad y digital prices
sin diferencias finitas):

    let z = (k - m), d = sqrt(z^2 + sigma^2)
    w'(k)  = b * (rho + z / d)
    w''(k) = b * sigma^2 / d^3
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.a, self.b, self.rho, self.m, self.sigma)


def total_variance(k: np.ndarray, p: SVIParams) -> np.ndarray:
    z = k - p.m
    d = np.sqrt(z * z + p.sigma * p.sigma)
    return p.a + p.b * (p.rho * z + d)


def total_variance_and_derivs(k: np.ndarray, p: SVIParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = k - p.m
    d = np.sqrt(z * z + p.sigma * p.sigma)
    w = p.a + p.b * (p.rho * z + d)
    wp = p.b * (p.rho + z / d)
    wpp = p.b * (p.sigma * p.sigma) / (d * d * d)
    return w, wp, wpp


def implied_vol_from_params(k: np.ndarray, p: SVIParams, T: float) -> np.ndarray:
    w = total_variance(np.atleast_1d(k), p)
    return np.sqrt(np.maximum(w / T, 1e-12))


def fit_svi(
    k_obs: np.ndarray,
    w_obs: np.ndarray,
    weights: np.ndarray | None = None,
    init: SVIParams | None = None,
) -> tuple[SVIParams, dict]:
    """Calibra SVI raw a (k_obs, w_obs) por minimos cuadrados ponderados.
    Devuelve (params, info_dict).
    """
    k = np.asarray(k_obs, dtype=float)
    w = np.asarray(w_obs, dtype=float)
    if weights is None:
        weights = np.ones_like(w)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(k) & np.isfinite(w) & (w > 0) & np.isfinite(weights) & (weights > 0)
    k = k[valid]
    w = w[valid]
    weights = weights[valid]
    if len(k) < 5:
        raise ValueError(f"need >= 5 valid points to fit SVI, got {len(k)}")

    if init is None:
        a0 = float(np.min(w))
        b0 = max(float(np.std(w)), 1e-4)
        init = SVIParams(a=a0, b=b0, rho=-0.1, m=0.0, sigma=0.1)

    def residuals(x):
        p = SVIParams(*x)
        return weights * (total_variance(k, p) - w)

    bounds = (
        [-1.0, 1e-6, -0.999, -2.0, 1e-4],   # a, b, rho, m, sigma
        [ 1.0, 5.0,   0.999,  2.0, 5.0  ],
    )
    res = least_squares(residuals, x0=list(init.as_tuple()), bounds=bounds,
                        max_nfev=5000, method="trf")
    p = SVIParams(*map(float, res.x))
    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "cost": float(res.cost),
        "n_points": int(len(k)),
        "n_iter": int(res.nfev),
    }
    return p, info


def bootstrap_svi_params(
    k_obs: np.ndarray,
    w_obs: np.ndarray,
    weights: np.ndarray | None = None,
    n_boot: int = 100,
    seed: int = 42,
) -> list[SVIParams]:
    """Genera N_boot fits SVI por re-muestreo con reemplazo (bootstrap paramétrico).

    Devuelve una lista de SVIParams (puede ser mas corta que n_boot si algunos
    fits fallan). Los params sirven para calcular intervalos de confianza de
    q_deribit downstream.

    Uso tipico:
        boot_params = bootstrap_svi_params(k, w, weights, n_boot=100)
        # Para cada bin [L, U]:
        q_boots = [range_prob_open_ended(L, U, ..., p=p) for p in boot_params]
        q_p05, q_p95 = np.percentile(q_boots, [5, 95])
    """
    k = np.asarray(k_obs, dtype=float)
    w = np.asarray(w_obs, dtype=float)
    if weights is None:
        weights = np.ones_like(w)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(k) & np.isfinite(w) & (w > 0) & np.isfinite(weights) & (weights > 0)
    k = k[valid]
    w = w[valid]
    weights = weights[valid]

    if len(k) < 5:
        return []

    rng = np.random.default_rng(seed)
    results: list[SVIParams] = []
    # Primero fijar el fit central como semilla inicial para cada bootstrap.
    try:
        p0, _ = fit_svi(k, w, weights)
    except Exception:
        return []

    n = len(k)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        kb, wb, wtsb = k[idx], w[idx], weights[idx]
        try:
            p, info = fit_svi(kb, wb, weights=wtsb, init=p0)
            if info.get("cost", float("inf")) < 1.0:   # filtro coste razonable
                results.append(p)
        except Exception:
            continue
    return results


def forward_from_pcp(
    strikes: np.ndarray,
    call_mids: np.ndarray,
    put_mids: np.ndarray,
    T: float,
    r: float = 0.0,
    spot_hint: float | None = None,
) -> float:
    """Forward implicito por put-call parity:
        F = K + e^(rT) * (C - P)
    Promedia sobre los pares mas cercanos al ATM (mas estables)."""
    K = np.asarray(strikes, dtype=float)
    C = np.asarray(call_mids, dtype=float)
    P = np.asarray(put_mids, dtype=float)
    valid = np.isfinite(K) & np.isfinite(C) & np.isfinite(P) & (C > 0) & (P > 0)
    K, C, P = K[valid], C[valid], P[valid]
    if len(K) == 0:
        if spot_hint is not None:
            return float(spot_hint)
        raise ValueError("no valid C/P pairs for PCP forward and no spot_hint")
    F_candidates = K + np.exp(r * T) * (C - P)
    if spot_hint is not None:
        order = np.argsort(np.abs(K - spot_hint))
        F_candidates = F_candidates[order][: max(1, min(5, len(F_candidates)))]
    return float(np.median(F_candidates))
