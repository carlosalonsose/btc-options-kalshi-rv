"""
Senal por bin: edge = lado-Kalshi - lado-Deribit.

Definiciones:
    edge_mid       = yes_mid - q_deribit          (firma del mispricing en mids)
    edge_buy_yes   = q_deribit - yes_ask          (>0 => YES Kalshi infravalorado vs Deribit)
    edge_sell_yes  = yes_bid  - q_deribit         (>0 => YES Kalshi sobrevalorado vs Deribit)

Por construccion sum(q_deribit) ~ 1 sobre un strip particion completo, asi que
cualquier edge_mid neto agregado del strip es ~0. La senal explotable vive en
la dispersion cross-sectional, no en el agregado.
"""
from __future__ import annotations

import pandas as pd

from src.model.exec_price import TwoSidedSVI, q_executable_bounds
from src.model.intraday_q import BasisModel, range_prob_open_ended
from src.model.svi import SVIParams


def annotate_edges(
    bins: pd.DataFrame,
    F: float,
    T_K: float,
    T_D: float,
    svi_params: SVIParams,
    basis: BasisModel | None = None,
    two_sided: TwoSidedSVI | None = None,
) -> pd.DataFrame:
    """Anade columnas q_deribit + edges al DataFrame de bins Kalshi.

    Si se pasa `two_sided` (3 SVI fits: bid/mid/ask), tambien anade:
        q_buy_exec, q_sell_exec
        edge_buy_yes_exec  = q_sell_exec - yes_ask  (>0 => arb estatico)
        edge_sell_yes_exec = yes_bid - q_buy_exec
    Estos son los edges que de verdad se pueden tradear (modulo Delta-hedge
    del mismatch T_K vs T_D)."""
    df = bins.copy()
    df["q_deribit"] = df.apply(
        lambda row: range_prob_open_ended(
            row.get("lower"), row.get("upper"),
            F=F, T_K=T_K, T_D=T_D, p=svi_params, basis=basis,
        ),
        axis=1,
    )
    if "yes_mid" in df.columns:
        df["edge_mid"] = df["yes_mid"] - df["q_deribit"]
    if "yes_ask" in df.columns:
        df["edge_buy_yes"] = df["q_deribit"] - df["yes_ask"]
    if "yes_bid" in df.columns:
        df["edge_sell_yes"] = df["yes_bid"] - df["q_deribit"]

    if two_sided is not None:
        bounds = df.apply(
            lambda row: q_executable_bounds(
                row.get("lower"), row.get("upper"),
                F=F, T_K=T_K, T_D=T_D, fit=two_sided, basis=basis,
            ),
            axis=1,
            result_type="expand",
        )
        bounds.columns = ["_q_mid_check", "q_buy_exec", "q_sell_exec"]
        df["q_buy_exec"] = bounds["q_buy_exec"].astype(float)
        df["q_sell_exec"] = bounds["q_sell_exec"].astype(float)
        if "yes_ask" in df.columns:
            df["edge_buy_yes_exec"] = df["q_sell_exec"] - df["yes_ask"]
        if "yes_bid" in df.columns:
            df["edge_sell_yes_exec"] = df["yes_bid"] - df["q_buy_exec"]
    return df


def basket_summary(annotated: pd.DataFrame) -> dict:
    """Diagnostico agregado: que tan bien suma a 1 la Q vs los precios Kalshi."""
    out = {
        "n_bins": int(len(annotated)),
        "sum_q_deribit": float(annotated["q_deribit"].sum()) if "q_deribit" in annotated else None,
    }
    for col in ["yes_mid", "yes_bid", "yes_ask"]:
        if col in annotated.columns:
            out[f"sum_{col}"] = float(annotated[col].sum(skipna=True))
    if "edge_mid" in annotated.columns:
        out["mean_abs_edge_mid"] = float(annotated["edge_mid"].abs().mean(skipna=True))
        out["max_abs_edge_mid"] = float(annotated["edge_mid"].abs().max(skipna=True))
    return out
