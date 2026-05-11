"""
Q intradia: dado un fit SVI a la smile Deribit @ T_D, calcula la probabilidad
neutral al riesgo de que el subyacente caiga en [L, U] al tiempo intradia T_K
(con T_K < T_D).

Asuncion central: VLT (variance lineal en tiempo).
    sigma_imp(K) en T_K = sigma_imp(K) en T_D = sqrt(w_D(k) / T_D)
    => P^Q(S_T_K > K) = N(d2) bajo BS con esa sigma y T_K.

Asuncion opcional: basis Gaussiano BRTI vs Deribit Index:
    S_T^BRTI = S_T^Deribit + epsilon, epsilon ~ N(mu_basis, sigma_basis^2 * T_K)
=> convolucion con kernel normal => engorda colas y desplaza media.
   En el limite mu=0, sigma=0 esto es identidad.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_digital_call
from .svi import SVIParams, total_variance


@dataclass(frozen=True)
class BasisModel:
    """Diferencia BRTI - Deribit_Index modelada como Gaussiano centrado.
    mu y sigma son por unidad de tiempo (var ~ sigma^2 * T)."""
    mu_per_year: float = 0.0
    sigma_per_sqrt_year: float = 0.0


def implied_sigma_at_strike(K: float, F: float, T_D: float, p: SVIParams) -> float:
    """sigma implied a partir de SVI raw, evaluada en K. Usa F como referencia."""
    if K <= 0 or F <= 0 or T_D <= 0:
        return float("nan")
    k = math.log(K / F)
    w = float(total_variance(np.array([k]), p)[0])
    return math.sqrt(max(w / T_D, 1e-12))


def survival_prob_intraday(K: float, F: float, T_K: float, T_D: float,
                           p: SVIParams, basis: BasisModel | None = None) -> float:
    """P^Q(S_T_K > K) bajo VLT scaling, con basis opcional."""
    sigma = implied_sigma_at_strike(K, F, T_D, p)
    if not (sigma > 0 and T_K > 0):
        return 0.0 if F < K else 1.0

    if basis is None or basis.sigma_per_sqrt_year == 0.0:
        # caso simple: BS digital sobre Deribit Index
        # ajustamos por mu_basis si lo hay (desplazamiento determinista del strike)
        mu = (basis.mu_per_year * T_K) if basis is not None else 0.0
        return bs_digital_call(F + mu, K, T_K, sigma, r=0.0)

    # con basis estocastico: convolucion analitica
    # S_T^BRTI = S_T^Deribit + epsilon
    # bajo log-normal para Deribit y normal para epsilon, no hay forma cerrada limpia.
    # usamos aproximacion: variance total = (F * sigma)^2 * T_K + basis.sigma^2 * T_K
    # y trabajamos con desviacion total efectiva en log-space.
    var_deribit = (sigma * sigma) * T_K
    var_basis_log = (basis.sigma_per_sqrt_year ** 2) * T_K / (F * F)
    sigma_eff = math.sqrt(var_deribit + var_basis_log)
    mu_eff = basis.mu_per_year * T_K
    d2 = (math.log((F + mu_eff) / K) - 0.5 * sigma_eff * sigma_eff) / sigma_eff
    return float(norm.cdf(d2))


def range_prob_intraday(L: float, U: float, F: float, T_K: float, T_D: float,
                        p: SVIParams, basis: BasisModel | None = None) -> float:
    """P^Q(L <= S_T_K <= U)."""
    if not (U > L):
        return 0.0
    L_eff = max(L, 1e-9)
    U_eff = max(U, L_eff + 1e-9)
    q_low = survival_prob_intraday(L_eff, F, T_K, T_D, p, basis)
    q_high = survival_prob_intraday(U_eff, F, T_K, T_D, p, basis)
    return float(np.clip(q_low - q_high, 0.0, 1.0))


def range_prob_open_ended(lower: float | None, upper: float | None,
                          F: float, T_K: float, T_D: float,
                          p: SVIParams, basis: BasisModel | None = None) -> float:
    """Maneja bins abiertos por arriba o por abajo (lower o upper = None / NaN).
    Convencion Kalshi: bin extremo inferior tiene cap_strike pero floor_strike puede
    interpretarse como -inf; idem para el extremo superior."""
    has_low = lower is not None and not (isinstance(lower, float) and math.isnan(lower))
    has_high = upper is not None and not (isinstance(upper, float) and math.isnan(upper))
    if has_low and has_high:
        return range_prob_intraday(float(lower), float(upper), F, T_K, T_D, p, basis)
    if has_high and not has_low:
        # P(S < U) = 1 - P(S > U)
        q_high = survival_prob_intraday(float(upper), F, T_K, T_D, p, basis)
        return float(np.clip(1.0 - q_high, 0.0, 1.0))
    if has_low and not has_high:
        # P(S > L)
        return float(np.clip(survival_prob_intraday(float(lower), F, T_K, T_D, p, basis), 0.0, 1.0))
    return 1.0
