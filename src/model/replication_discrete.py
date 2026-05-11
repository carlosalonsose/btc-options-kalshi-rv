"""
Replicacion discreta del bin Kalshi con strikes Deribit reales.

Crítica que motiva este modulo:

    El bin Kalshi paga $1 si BTC esta en [L, U] (rectangulo binario).
    La Q teorica P(L<=S<=U) calculada via SVI digital asume strikes
    continuos. Pero Deribit cotiza strikes discretos cada $500-$1000,
    mientras que los bins Kalshi son de $100. Eso impide replicar el
    payoff rectangular exactamente.

Lo que SI se puede construir en Deribit (con strikes K_low <= L < U <= K_high):

    Vertical de calls (long K_low, short K_high):
       payoff_per_BTC(S) = max(0, min(S - K_low, K_high - K_low))

    Es decir, una rampa lineal entre K_low y K_high, no un rectangulo.

Modelo de hedge realista (proportional-width):

    Para hedgear `n` contratos Kalshi de ancho (U - L), compramos
    `n * (U - L) / (K_high - K_low)` verticales. Eso da exposicion
    nocional correcta en promedio bajo Q, dejando un tracking error
    que se gestionaria con Delta-hedge dinamico hasta T_K.

    Coste neto por 1 contrato Kalshi cubierto:

        q_buy_repl_disc  = (U-L)/(K_high-K_low) * vertical_ask_cost
        q_sell_repl_disc = (U-L)/(K_high-K_low) * vertical_bid_cost

    donde:
        vertical_ask_cost = call(K_low)_ask - call(K_high)_bid   (compras)
        vertical_bid_cost = call(K_low)_bid - call(K_high)_ask   (vendes)

Esta es la version mas honesta del precio ejecutable del bin Kalshi:
    q_sell_repl_disc <= q_buy_repl_disc  (siempre, modulo arbitraje)
    Comparar yes_bid_kalshi vs q_buy_repl_disc da el edge SELL YES real.

Limitaciones (no resueltas en V1):
- Tracking error de payoff residual no modelado en PnL realizado.
- No usa puts (solo calls); para bins muy OTM se podrian preferir puts.
- Asume que el hedge se mantiene hasta T_K sin re-balanceo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VerticalQuote:
    K_low: float
    K_high: float
    call_low_bid: float
    call_low_ask: float
    call_high_bid: float
    call_high_ask: float

    @property
    def vertical_ask_cost(self) -> float:
        """Coste de COMPRAR la vertical: pagas ask del low, cobras bid del high."""
        return float(self.call_low_ask - self.call_high_bid)

    @property
    def vertical_bid_cost(self) -> float:
        """Coste de VENDER la vertical: cobras bid del low, pagas ask del high.
        Si es positivo, recibes este monto al vender la vertical."""
        return float(self.call_low_bid - self.call_high_ask)

    @property
    def width(self) -> float:
        return float(self.K_high - self.K_low)


def find_bracketing_strikes(L: float | None, U: float | None,
                            strikes: np.ndarray) -> tuple[float, float] | None:
    """Encuentra K_low (mayor strike <= L) y K_high (menor strike >= U)
    en el array de strikes disponibles. Devuelve None si no hay bracket
    (ej. bin tail abierto fuera del rango Deribit)."""
    if L is None or U is None or not math.isfinite(L) or not math.isfinite(U):
        return None
    s = np.asarray(strikes, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    s = np.sort(np.unique(s))
    below = s[s <= L]
    above = s[s >= U]
    if below.size == 0 or above.size == 0:
        return None
    return float(below[-1]), float(above[0])


def get_call_quotes(K: float, calls_df: pd.DataFrame) -> tuple[float, float] | None:
    """Devuelve (bid, ask) de la call al strike K, o None si no hay quotes
    validos (bid <= 0 o ask <= 0 o NaN)."""
    row = calls_df[calls_df["strike"] == K]
    if row.empty:
        return None
    bid = float(row["bid_price"].iloc[0])
    ask = float(row["ask_price"].iloc[0])
    if not (math.isfinite(bid) and math.isfinite(ask) and bid > 0 and ask > 0):
        return None
    return bid, ask


def replicate_bin(lower: float | None, upper: float | None,
                  calls_df: pd.DataFrame) -> dict | None:
    """Construye la replicacion discreta para un bin Kalshi [lower, upper]
    usando las calls disponibles en calls_df (1 expiry, columnas strike,
    bid_price, ask_price). Devuelve None si no hay bracket con quotes validos.

    calls_df debe estar pre-filtrado a una sola expiry.
    """
    if lower is None or upper is None:
        return None
    if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
        return None

    bracket = find_bracketing_strikes(lower, upper, calls_df["strike"].values)
    if bracket is None:
        return None
    K_low, K_high = bracket
    if K_high <= K_low:
        return None

    q_low = get_call_quotes(K_low, calls_df)
    q_high = get_call_quotes(K_high, calls_df)
    if q_low is None or q_high is None:
        return None

    vq = VerticalQuote(
        K_low=K_low, K_high=K_high,
        call_low_bid=q_low[0], call_low_ask=q_low[1],
        call_high_bid=q_high[0], call_high_ask=q_high[1],
    )
    width_kalshi = upper - lower
    width_vertical = vq.width
    ratio = width_kalshi / width_vertical  # 0 < ratio <= 1 typically

    # Conversion correcta: la vertical de calls cuesta C en USD y tiene
    # payoff maximo (K_high - K_low) USD. La "probabilidad-equivalente"
    # del rango ampliado es C / width_vertical (entre 0 y 1).
    # Para escalar a la probabilidad del bin Kalshi (ancho width_kalshi)
    # asumiendo densidad localmente uniforme en [K_low, K_high]:
    #     q_kalshi_implicita = (width_kalshi/width_vertical) * (C/width_vertical)
    #                        = ratio * (C/width_vertical)
    prob_range_buy = vq.vertical_ask_cost / width_vertical
    prob_range_sell = vq.vertical_bid_cost / width_vertical
    q_buy_repl = ratio * prob_range_buy
    q_sell_repl = ratio * prob_range_sell

    # Clip para mantenerlas en [0, 1] como probabilidades. Numericamente
    # prob_range_sell puede salir < 0 (si call_low_bid < call_high_ask),
    # en cuyo caso no hay precio ejecutable razonable para vender la
    # vertical → marca como NaN para que no se use.
    if not math.isfinite(q_sell_repl) or q_sell_repl < 0:
        q_sell_repl = float("nan")
    if not math.isfinite(q_buy_repl) or q_buy_repl < 0 or q_buy_repl > 1:
        q_buy_repl = float("nan")

    return {
        "K_low": K_low,
        "K_high": K_high,
        "width_kalshi": width_kalshi,
        "width_vertical": width_vertical,
        "ratio_widths": ratio,
        "vertical_ask_cost": vq.vertical_ask_cost,
        "vertical_bid_cost": vq.vertical_bid_cost,
        "prob_range_buy": prob_range_buy,
        "prob_range_sell": prob_range_sell,
        "q_buy_repl_disc": q_buy_repl,
        "q_sell_repl_disc": q_sell_repl,
    }


def annotate_replication(bins: pd.DataFrame, calls_df: pd.DataFrame) -> pd.DataFrame:
    """Anade columnas de replicacion discreta a un DataFrame de bins Kalshi.
    calls_df: chain Deribit filtrada a una sola expiry, solo calls."""
    df = bins.copy()
    out = df.apply(
        lambda row: replicate_bin(row.get("lower"), row.get("upper"), calls_df),
        axis=1,
    )
    cols = ["K_low", "K_high", "width_kalshi", "width_vertical", "ratio_widths",
            "vertical_ask_cost", "vertical_bid_cost",
            "prob_range_buy", "prob_range_sell",
            "q_buy_repl_disc", "q_sell_repl_disc"]
    for c in cols:
        df[c] = out.apply(lambda d: d[c] if d is not None else float("nan"))
    return df
