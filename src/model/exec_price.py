"""
Precio Q ejecutable: replicación bid/ask via SVI side-fits.

Motivación: la Q calculada con mid_price de las opciones Deribit es teorica.
Para tradear realmente la replicacion en Deribit hay que cruzar el bid-ask de
cada opcion. Esto se aproxima fitteando dos SVIs adicionales:

  - svi_bid: a partir de bid_price -> iv_bid (cota inferior de la sigma).
  - svi_ask: a partir de ask_price -> iv_ask (cota superior de la sigma).

Para un bin [L, U]:
    q_under_bid = range_prob(L, U, svi_bid)
    q_under_ask = range_prob(L, U, svi_ask)
    q_buy_exec  = max(q_under_bid, q_under_ask)   <-- pagas para estar LARGO
    q_sell_exec = min(q_under_bid, q_under_ask)   <-- cobras para estar CORTO

Estas son cotas que ENVUELVEN q_mid (modulo errores de fit). Conservadoras en
ambas direcciones. Es una aproximacion al precio replicable: una version mas
rigurosa requeriria construir verticales con strikes Deribit discretos, pero
captura el efecto principal (bid-ask wide en Deribit -> Q ejecutable difiere
mucho de Q teorica) sin sobrecargar la pipeline.

Para cada bin se exponen luego dos edges ejecutables:
    edge_buy_yes_exec  = q_sell_exec - yes_ask   (>0 => buy YES Kalshi + short replicacion Deribit)
    edge_sell_yes_exec = yes_bid - q_buy_exec    (>0 => sell YES Kalshi + long replicacion Deribit)

Si edge_*_exec < 0 a thresholds razonables, no hay arb estatico que sobreviva
el bid-ask Deribit. (No incluye mismatch T_K vs T_D ni basis BRTI - eso lo
gestiona Delta-hedge en live.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import implied_vol
from .intraday_q import BasisModel, range_prob_open_ended
from .svi import SVIParams, fit_svi


@dataclass(frozen=True)
class TwoSidedSVI:
    """Tres fits SVI sobre la misma chain: bid-side, mid (default), ask-side."""
    mid: SVIParams
    bid: SVIParams | None
    ask: SVIParams | None
    info_mid: dict
    info_bid: dict | None
    info_ask: dict | None
    n_points_mid: int
    n_points_bid: int
    n_points_ask: int


def _iv_for_side(row: pd.Series, side: str, S: float, T_D: float) -> float:
    """Calcula iv usando bid_price o ask_price segun side. Devuelve nan si invalida."""
    px = row.get(f"{side}_price")
    if px is None or not np.isfinite(px) or px <= 0:
        return np.nan
    K = row.get("strike")
    typ = "C" if row.get("option_type") == "call" else "P"
    return implied_vol(float(px), S, float(K), T_D, typ)


def compute_iv_bid_mid_ask(chain_e: pd.DataFrame, F: float, T_D: float) -> pd.DataFrame:
    """Para una chain de una sola expiry (ya filtrada), devuelve un DataFrame
    OTM-only con columnas iv_bid, iv_mid, iv_ask (NaN cuando no se puede invertir)."""
    df = chain_e.copy()
    df = df[(df["bid_price"] > 0) & (df["ask_price"] > 0) & (df["mid_price"] > 0)].copy()
    if df.empty:
        return df

    spot = float(df["underlying_price"].dropna().median())

    calls = df[df["option_type"] == "call"].copy()
    puts = df[df["option_type"] == "put"].copy()
    otm_calls = calls[calls["strike"] >= F].copy()
    otm_puts = puts[puts["strike"] < F].copy()
    otm = pd.concat([otm_calls, otm_puts], ignore_index=True)
    if otm.empty:
        return otm

    otm["iv_bid"] = otm.apply(lambda r: _iv_for_side(r, "bid", spot, T_D), axis=1)
    otm["iv_mid"] = otm.apply(lambda r: _iv_for_side(r, "mid", spot, T_D), axis=1)
    otm["iv_ask"] = otm.apply(lambda r: _iv_for_side(r, "ask", spot, T_D), axis=1)
    otm["k"] = np.log(otm["strike"] / F)
    otm["spread"] = otm["ask_price"] - otm["bid_price"]
    return otm


def fit_two_sided_svi(otm_df: pd.DataFrame, T_D: float) -> TwoSidedSVI:
    """Fitea SVI 3 veces (bid, mid, ask) reusando la misma chain OTM. Si bid o
    ask no tienen suficientes puntos validos, devuelven None y se usa mid como
    fallback en consumidores."""
    weights_full = 1.0 / np.maximum(otm_df["spread"].fillna(otm_df["spread"].median()).values, 1e-3)

    def _fit(col):
        valid = otm_df[col].notna() & (otm_df[col] > 0)
        sub = otm_df[valid]
        if len(sub) < 5:
            return None, {"success": False, "n_points": int(len(sub)),
                          "message": "insuficientes IVs", "cost": float("nan")}
        k = sub["k"].values
        w = (sub[col].values ** 2) * T_D
        wts = weights_full[valid.values]
        try:
            params, info = fit_svi(k, w, weights=wts)
            return params, info
        except Exception as e:
            return None, {"success": False, "n_points": int(len(sub)),
                          "message": str(e), "cost": float("nan")}

    p_mid, info_mid = _fit("iv_mid")
    if p_mid is None:
        raise ValueError(f"fit SVI mid fallo: {info_mid.get('message')}")
    p_bid, info_bid = _fit("iv_bid")
    p_ask, info_ask = _fit("iv_ask")

    return TwoSidedSVI(
        mid=p_mid, bid=p_bid, ask=p_ask,
        info_mid=info_mid, info_bid=info_bid, info_ask=info_ask,
        n_points_mid=int(info_mid.get("n_points", 0)),
        n_points_bid=int(info_bid.get("n_points", 0)) if info_bid else 0,
        n_points_ask=int(info_ask.get("n_points", 0)) if info_ask else 0,
    )


def q_executable_bounds(
    lower: float | None,
    upper: float | None,
    F: float,
    T_K: float,
    T_D: float,
    fit: TwoSidedSVI,
    basis: BasisModel | None = None,
) -> tuple[float, float, float]:
    """Devuelve (q_mid, q_buy_exec, q_sell_exec) para el bin [lower, upper].

    q_buy_exec  = max(q_under_bid, q_under_ask)  -- pagas para LONG
    q_sell_exec = min(q_under_bid, q_under_ask)  -- cobras para SHORT
    q_mid       = q bajo svi_mid (referencia, no ejecutable).

    Si bid o ask SVI no convergieron, fallback a mid -> bounds colapsan a q_mid.
    """
    q_mid = range_prob_open_ended(lower, upper, F=F, T_K=T_K, T_D=T_D,
                                  p=fit.mid, basis=basis)
    q_b = (range_prob_open_ended(lower, upper, F=F, T_K=T_K, T_D=T_D,
                                 p=fit.bid, basis=basis)
           if fit.bid is not None else q_mid)
    q_a = (range_prob_open_ended(lower, upper, F=F, T_K=T_K, T_D=T_D,
                                 p=fit.ask, basis=basis)
           if fit.ask is not None else q_mid)
    q_buy = max(q_b, q_a)
    q_sell = min(q_b, q_a)
    return float(q_mid), float(q_buy), float(q_sell)
