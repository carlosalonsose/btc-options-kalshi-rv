"""
Expiry quality scoring — Sprint 2.

Filtra expiries de Deribit con pocos quotes validos (mercado cerrado, cercano
a vencimiento, overnight baja liquidez). Una expiry con calidad baja produce
fits SVI inestables y q_deribit poco fiables.

Criterios:
  - n_valid_otm : strikes OTM con bid > 0 AND ask > 0
  - n_valid_iv  : de los anteriores, cuantos convergen a IV real (>0, finito)
  - score       = n_valid_iv / max(n_total_otm_possible, 1)    in [0, 1]

Thresholds recomendados:
  - score >= 0.25  (al menos 25% de strikes liquidos)  → calidad baja pero usable
  - score >= 0.50  → calidad media
  - score >= 0.75  → calidad alta

Uso en analyze.py:
    from src.model.quality import expiry_quality_score
    quality = expiry_quality_score(chain_e, F=F, T_D=T_D)
    # quality.score, quality.n_valid_iv, quality.n_total_otm, quality.ok
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .black_scholes import implied_vol


@dataclass(frozen=True)
class ExpiryQuality:
    score: float          # 0 = no quotes usables, 1 = todos los strikes cotizados y válidos
    n_total_otm: int      # strikes OTM en la chain (calls K>=F, puts K<F)
    n_quoted: int         # con bid>0 AND ask>0
    n_valid_iv: int       # con IV inversa convergente
    ok: bool              # True si score >= MIN_SCORE y n_valid_iv >= MIN_STRIKES

    # Mensaje legible para logs/CLI
    message: str

    MIN_SCORE: float = 0.25
    MIN_STRIKES: int = 5

    @classmethod
    def bad(cls, reason: str) -> "ExpiryQuality":
        return cls(score=0.0, n_total_otm=0, n_quoted=0, n_valid_iv=0,
                   ok=False, message=reason)


def expiry_quality_score(
    chain_e: pd.DataFrame,
    F: float,
    T_D: float,
    min_score: float = ExpiryQuality.MIN_SCORE,
    min_strikes: int = ExpiryQuality.MIN_STRIKES,
) -> ExpiryQuality:
    """Evalua la calidad de la surface Deribit para una expiry.

    Parameters
    ----------
    chain_e   : DataFrame de opciones para UNA expiry (ya filtrado con bid>0 y ask>0).
    F         : forward price del PCP.
    T_D       : tiempo hasta expiry en años.
    min_score : umbral de score para marcar ok=True.
    min_strikes: numero minimo de strikes con IV valida para marcar ok=True.
    """
    if chain_e is None or chain_e.empty:
        return ExpiryQuality.bad("chain_e vacia")

    # Todos los strikes presentes (con cualquier quote)
    calls_all = chain_e[chain_e["option_type"] == "call"]
    puts_all = chain_e[chain_e["option_type"] == "put"]
    n_calls_otm = int((calls_all["strike"] >= F).sum())
    n_puts_otm = int((puts_all["strike"] < F).sum())
    n_total_otm = n_calls_otm + n_puts_otm

    if n_total_otm == 0:
        return ExpiryQuality.bad("sin strikes OTM")

    # Filtrar solo con bid>0 AND ask>0
    quoted = chain_e[(chain_e["bid_price"] > 0) & (chain_e["ask_price"] > 0)].copy()
    otm_quoted = pd.concat([
        quoted[(quoted["option_type"] == "call") & (quoted["strike"] >= F)],
        quoted[(quoted["option_type"] == "put") & (quoted["strike"] < F)],
    ], ignore_index=True)
    n_quoted = len(otm_quoted)

    if n_quoted == 0:
        return ExpiryQuality.bad("sin OTM con bid+ask > 0")

    spot = float(chain_e["underlying_price"].dropna().median())
    if not math.isfinite(spot) or spot <= 0:
        spot = F  # fallback

    # Intentar IV inversion para cada OTM cotizado
    n_valid_iv = 0
    for _, row in otm_quoted.iterrows():
        try:
            px = float(row.get("mid_price") or 0)
            K = float(row.get("strike") or 0)
            typ = "C" if row.get("option_type") == "call" else "P"
            if px <= 0 or K <= 0:
                continue
            iv = implied_vol(px, spot, K, T_D, typ)
            if iv is not None and math.isfinite(iv) and iv > 0.01:
                n_valid_iv += 1
        except Exception:
            continue

    score = n_valid_iv / max(n_total_otm, 1)
    ok = (score >= min_score) and (n_valid_iv >= min_strikes)

    msg_parts = [
        f"n_valid_iv={n_valid_iv}/{n_total_otm}",
        f"score={score:.2f}",
    ]
    if not ok:
        msg_parts.append(f"BELOW_THRESHOLD(min={min_score:.2f},min_k={min_strikes})")
    message = "  ".join(msg_parts)

    return ExpiryQuality(
        score=score,
        n_total_otm=n_total_otm,
        n_quoted=n_quoted,
        n_valid_iv=n_valid_iv,
        ok=ok,
        message=message,
    )
