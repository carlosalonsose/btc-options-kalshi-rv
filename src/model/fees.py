"""
Modelo de fees Kalshi + Deribit.

Kalshi (binary markets, formula oficial):
    fee_per_contract = ceil( 0.07 * price * (1 - price), 0.01 )
    Solo se cobra al fill. Settlement y expiry no llevan fee adicional.

Deribit (opciones BTC):
    fee = min( 0.03% * notional, 12.5% * premium )
    Para verticales ATM con premium bajo el cap de 12.5% suele activarse.
    Para nuestro proposito de backtest aproximamos con un valor por contrato
    constante calibrable; el detalle exacto requeriria conocer notional/premium
    de cada pata, mejorable en futuras iteraciones.

Para la estrategia "SELL YES Kalshi + LONG vertical Deribit":
    fee_total_per_contract = fee_kalshi(yes_bid) + fee_deribit_constant
"""
from __future__ import annotations

import math


def kalshi_fee(price: float) -> float:
    """Fee Kalshi por contrato al precio dado, redondeado hacia arriba al centavo.

    Formula oficial:
        ceil( 0.07 * price * (1-price), 0.01 )
    """
    if price is None or not math.isfinite(price) or price <= 0 or price >= 1:
        return 0.0
    raw = 0.07 * price * (1.0 - price)
    return math.ceil(raw * 100.0) / 100.0


def deribit_fee_per_contract(default: float = 0.015) -> float:
    """Fee Deribit aproximado por contrato Kalshi cubierto.

    Aproximacion pragmatica: en la replicacion 1:1 de un bin Kalshi via
    una vertical estrecha de calls Deribit, los fees agregados de las
    dos patas (entry compra L, entry venta U) suelen estar en el rango
    1-2 cts por contrato Kalshi equivalente.

    Default 0.015 USD. Configurable desde el backtest/Streamlit.
    """
    return float(default)


def total_fee_sell_yes(yes_bid: float,
                      deribit_fee_const: float = 0.015) -> float:
    """Fees totales para 1 contrato de la estrategia SELL YES + LONG vertical."""
    return kalshi_fee(yes_bid) + deribit_fee_per_contract(deribit_fee_const)


def total_fee_buy_yes(yes_ask: float,
                     deribit_fee_const: float = 0.015) -> float:
    """Fees totales para 1 contrato de la estrategia BUY YES + SHORT vertical."""
    return kalshi_fee(yes_ask) + deribit_fee_per_contract(deribit_fee_const)
