"""
Black-Scholes basico: precio call/put y solver de implied volatility.
Asume tasa cero por defecto (apropiado para horizontes cripto cortos).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 0.0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        return max(S - K * math.exp(-r * T), 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    return bs_call(S, K, T, sigma, r) - S + K * math.exp(-r * T)


def bs_digital_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """P^Q(S_T > K) bajo BS. Devuelve N(d2)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0 if S < K else 1.0
    d2 = (math.log(S / K) + (r - 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(d2))


def implied_vol(price: float, S: float, K: float, T: float,
                option_type: str, r: float = 0.0) -> float:
    """Resuelve sigma tal que BS price = market price. Devuelve nan si no es solvable."""
    if T <= 0.0 or price <= 0.0 or S <= 0.0 or K <= 0.0:
        return np.nan
    intr_call = max(S - K * math.exp(-r * T), 0.0)
    intr_put = max(K * math.exp(-r * T) - S, 0.0)
    if option_type.upper() in ("C", "CALL"):
        if price <= intr_call or price >= S:
            return np.nan
        f = lambda v: bs_call(S, K, T, v, r) - price
    else:
        if price <= intr_put or price >= K * math.exp(-r * T):
            return np.nan
        f = lambda v: bs_put(S, K, T, v, r) - price
    try:
        return float(brentq(f, 1e-6, 5.0))
    except ValueError:
        return np.nan
