"""
Pipeline end-to-end sobre un snapshot grabado:

  1. Carga el snapshot (default: el mas reciente).
  2. Elige el evento Kalshi mas cercano al ahora.
  3. Selecciona la expiry Deribit mas cercana posterior al settlement Kalshi.
  4. Limpia quotes, fitea SVI sobre la smile.
  5. Calcula Q intradia para cada bin Kalshi via VLT scaling.
  6. Imprime tabla de bins con kalshi_mid, q_deribit, edges.
  7. Guarda plot smile + Q (opcional).

Uso:
  .venv/bin/python -m src.runner.analyze
  .venv/bin/python -m src.runner.analyze --snapshot data/snapshots/2026-05-08/215438.json.gz
  .venv/bin/python -m src.runner.analyze --event KXBTC-26MAY0817 --plot
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.load import (
    deribit_btc_usdc_chain,
    kalshi_btc_range,
    kalshi_open_events,
    latest_snapshot,
    list_snapshots,
    load_snapshot,
    select_deribit_expiry,
    select_event_for_settle,
)
from src.model.black_scholes import implied_vol
from src.model.exec_price import compute_iv_bid_mid_ask, fit_two_sided_svi
from src.model.intraday_q import BasisModel
from src.model.replication_discrete import annotate_replication
from src.model.svi import SVIParams, fit_svi, forward_from_pcp, total_variance, implied_vol_from_params
from src.signal.edge import annotate_edges, basket_summary

YEAR_SECONDS = 365.0 * 24.0 * 3600.0


def _T_in_years(t_now: pd.Timestamp, t_target: pd.Timestamp) -> float:
    return max((t_target - t_now).total_seconds() / YEAR_SECONDS, 1e-7)


def _print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_pipeline(snap: dict, event_ticker: str | None = None) -> dict:
    """Pipeline puro: snap + (opcional) event_ticker -> dict con todos los resultados.
    No imprime nada, no toca disco, no arranca threads. Reutilizable desde el CLI
    y desde la UI."""
    snap_ts = pd.Timestamp(snap["snapshot_ts_utc"])
    if snap_ts.tzinfo is None:
        snap_ts = snap_ts.tz_localize("UTC")

    events = kalshi_open_events(snap)
    if event_ticker is None:
        event_ticker = select_event_for_settle(snap, snap_ts)
    bins = kalshi_btc_range(snap, event_ticker=event_ticker)
    if bins.empty:
        raise ValueError(f"event_ticker={event_ticker} sin bins en el snapshot")

    settle = pd.Timestamp(bins["expected_expiration_time"].iloc[0])
    if settle.tzinfo is None:
        settle = settle.tz_localize("UTC")
    T_K = _T_in_years(snap_ts, settle)

    chain = deribit_btc_usdc_chain(snap)
    if chain.empty:
        raise ValueError("Deribit chain vacia en el snapshot")
    expiry_dt = select_deribit_expiry(chain, settle)
    chain_e = chain[chain["expiry_dt"] == expiry_dt].copy()
    T_D = _T_in_years(snap_ts, expiry_dt)
    spot = float(chain_e["underlying_price"].dropna().median())

    chain_e = chain_e[
        (chain_e["bid_price"] > 0)
        & (chain_e["ask_price"] > 0)
        & (chain_e["mid_price"] > 0)
    ].copy()
    calls = chain_e[chain_e["option_type"] == "call"].copy()
    puts = chain_e[chain_e["option_type"] == "put"].copy()

    common_K = sorted(set(calls["strike"]).intersection(set(puts["strike"])))
    if common_K:
        c_mid = calls.set_index("strike")["mid_price"].groupby(level=0).first().reindex(common_K).values
        p_mid = puts.set_index("strike")["mid_price"].groupby(level=0).first().reindex(common_K).values
        F = forward_from_pcp(np.array(common_K, dtype=float), c_mid, p_mid, T=T_D, spot_hint=spot)
    else:
        F = spot

    otm_calls = calls[calls["strike"] >= F].copy()
    otm_calls["iv"] = otm_calls.apply(
        lambda r: implied_vol(r["mid_price"], spot, r["strike"], T_D, "C"), axis=1)
    otm_puts = puts[puts["strike"] < F].copy()
    otm_puts["iv"] = otm_puts.apply(
        lambda r: implied_vol(r["mid_price"], spot, r["strike"], T_D, "P"), axis=1)

    fit_df = pd.concat([otm_calls, otm_puts], ignore_index=True)
    fit_df = fit_df[fit_df["iv"].notna() & (fit_df["iv"] > 0)].copy()
    if len(fit_df) < 5:
        raise ValueError(f"insuficientes IVs validos para fit SVI: {len(fit_df)}")
    fit_df["k"] = np.log(fit_df["strike"] / F)
    fit_df["w"] = fit_df["iv"] ** 2 * T_D
    weights = 1.0 / np.maximum(fit_df["spread"].fillna(fit_df["spread"].median()).values, 1e-3)

    svi_params, info = fit_svi(fit_df["k"].values, fit_df["w"].values, weights=weights)

    # Fase 3.5: fits adicionales bid/ask para precio Q ejecutable.
    otm_full = compute_iv_bid_mid_ask(chain_e, F=F, T_D=T_D)
    try:
        two_sided = fit_two_sided_svi(otm_full, T_D=T_D)
    except Exception:
        two_sided = None

    annotated = annotate_edges(
        bins, F=F, T_K=T_K, T_D=T_D, svi_params=svi_params,
        basis=BasisModel(), two_sided=two_sided,
    )

    # Sprint 1.5: replicacion discreta con strikes Deribit reales.
    # Le pasa solo las CALLS de la chain Deribit (no puts, V1).
    calls_only = chain_e[chain_e["option_type"] == "call"].copy()
    annotated = annotate_replication(annotated, calls_only)
    if "yes_ask" in annotated.columns and "q_sell_repl_disc" in annotated.columns:
        annotated["edge_buy_yes_repl"] = annotated["q_sell_repl_disc"] - annotated["yes_ask"]
    if "yes_bid" in annotated.columns and "q_buy_repl_disc" in annotated.columns:
        annotated["edge_sell_yes_repl"] = annotated["yes_bid"] - annotated["q_buy_repl_disc"]

    summary = basket_summary(annotated)

    return {
        "snap_ts": snap_ts,
        "event_ticker": event_ticker,
        "events": events,
        "bins": bins,
        "settle": settle,
        "T_K": T_K,
        "expiry_dt": expiry_dt,
        "T_D": T_D,
        "spot": spot,
        "F": F,
        "fit_df": fit_df,
        "svi_params": svi_params,
        "svi_info": info,
        "two_sided": two_sided,
        "annotated": annotated,
        "summary": summary,
    }


def run(snapshot_path: Path | None, event_ticker: str | None,
        plot: bool, plot_dir: Path) -> None:
    snap = latest_snapshot() if snapshot_path is None else load_snapshot(snapshot_path)
    res = run_pipeline(snap, event_ticker=event_ticker)
    snap_ts = res["snap_ts"]
    event_ticker = res["event_ticker"]
    events = res["events"]
    bins = res["bins"]
    settle = res["settle"]
    T_K = res["T_K"]
    expiry_dt = res["expiry_dt"]
    T_D = res["T_D"]
    spot = res["spot"]
    F = res["F"]
    fit_df = res["fit_df"]
    svi_params = res["svi_params"]
    info = res["svi_info"]
    annotated = res["annotated"]

    _print_section("Snapshot")
    print(f"  ts_utc           = {snap_ts}")
    print(f"  source           = {snapshot_path or 'latest'}")

    _print_section("Eventos Kalshi abiertos en el snapshot")
    print(events.to_string(index=False))

    _print_section(f"Evento Kalshi seleccionado: {event_ticker}")
    print(f"  settlement_utc   = {settle}")
    print(f"  T_K (años)       = {T_K:.6e}  ({T_K * YEAR_SECONDS / 3600:.2f}h)")
    print(f"  n_bins           = {len(bins)}")

    _print_section(f"Deribit smile @ {expiry_dt.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  T_D (años)       = {T_D:.6e}  ({T_D * YEAR_SECONDS / 3600:.2f}h)")
    print(f"  spot (Deribit)   = {spot:,.2f}")
    print(f"  n_options        = {len(fit_df)} OTM puntos para fit")
    print(f"  forward (PCP)    = {F:,.2f}")

    _print_section("Fit SVI")
    print(f"  n_points         = {info['n_points']}")
    print(f"  cost             = {info['cost']:.6e}")
    print(f"  success          = {info['success']}")
    print(f"  params           = a={svi_params.a:.4f} b={svi_params.b:.4f} "
          f"rho={svi_params.rho:.4f} m={svi_params.m:.4f} sigma={svi_params.sigma:.4f}")

    # Filtramos bins relevantes: q_deribit > 0.005 OR yes_mid > 0.02
    relevance = (annotated.get("q_deribit", 0) > 0.005) | (annotated.get("yes_mid", 0) > 0.02)
    relevant = annotated[relevance].copy().sort_values(by=["lower"], na_position="first")

    _print_section(f"Bins relevantes (q>0.5% o yes_mid>2%): {len(relevant)} de {len(annotated)}")
    show_cols = [c for c in [
        "subtitle", "yes_bid", "yes_ask", "yes_mid",
        "q_deribit", "edge_mid", "edge_buy_yes", "edge_sell_yes",
    ] if c in relevant.columns]
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    pd.set_option("display.width", 160)
    print(relevant[show_cols].to_string(index=False))

    # Top oportunidades para comprar YES (precio Kalshi ask < Q deribit)
    if "edge_buy_yes" in annotated.columns:
        top_buy = annotated[annotated["edge_buy_yes"] > 0.01].copy()
        top_buy = top_buy.sort_values("edge_buy_yes", ascending=False).head(10)
        if not top_buy.empty:
            _print_section("Top oportunidades BUY YES (q_deribit > yes_ask + 1%)")
            print(top_buy[show_cols].to_string(index=False))

    # Top oportunidades para vender YES (precio Kalshi bid > Q deribit)
    if "edge_sell_yes" in annotated.columns:
        top_sell = annotated[annotated["edge_sell_yes"] > 0.01].copy()
        top_sell = top_sell.sort_values("edge_sell_yes", ascending=False).head(10)
        if not top_sell.empty:
            _print_section("Top oportunidades SELL YES (yes_bid > q_deribit + 1%)")
            print(top_sell[show_cols].to_string(index=False))

    summary = res["summary"]
    _print_section("Diagnostico agregado del strip")
    for k, v in summary.items():
        print(f"  {k:25s} = {v}")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            plot_dir.mkdir(parents=True, exist_ok=True)
            stamp = snap_ts.strftime("%Y%m%d_%H%M%S")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            k_grid = np.linspace(fit_df["k"].min() - 0.05, fit_df["k"].max() + 0.05, 300)
            iv_grid = implied_vol_from_params(k_grid, svi_params, T_D)
            axes[0].scatter(fit_df["k"], fit_df["iv"], alpha=0.7, label="market IV (calls)")
            axes[0].plot(k_grid, iv_grid, color="black", lw=2, label="SVI fit")
            axes[0].set_xlabel("log-moneyness k = ln(K/F)")
            axes[0].set_ylabel("Implied vol")
            axes[0].set_title(f"SVI smile  Deribit {expiry_dt.strftime('%Y-%m-%d %H:%MZ')}")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            edge_col = "edge_mid" if "edge_mid" in annotated.columns else "q_deribit"
            mid_strikes = ((annotated["lower"].fillna(annotated["upper"]) +
                            annotated["upper"].fillna(annotated["lower"])) / 2.0)
            colors = ["tab:red" if v > 0 else "tab:green" for v in annotated[edge_col].fillna(0)]
            axes[1].bar(mid_strikes, annotated[edge_col].fillna(0), width=200, color=colors, alpha=0.7)
            axes[1].axhline(0, color="black", lw=0.5)
            axes[1].set_xlabel("Bin midpoint (USD)")
            axes[1].set_ylabel(edge_col)
            axes[1].set_title(f"{event_ticker}  edge_mid = yes_mid - q_deribit")
            axes[1].grid(True, alpha=0.3)

            out_path = plot_dir / f"{stamp}_{event_ticker}.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
            print(f"\nplot guardado: {out_path}")
        except Exception as e:
            print(f"plot fallo: {e}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Analiza un snapshot end-to-end")
    p.add_argument("--snapshot", type=Path, default=None,
                   help="ruta a un snapshot .json.gz (default: el mas reciente)")
    p.add_argument("--event", type=str, default=None,
                   help="event_ticker Kalshi (default: el mas cercano al ahora)")
    p.add_argument("--plot", action="store_true",
                   help="genera plot smile + edges en data/reports/")
    p.add_argument("--plot-dir", type=Path, default=Path("data/reports"))
    args = p.parse_args(argv)
    run(args.snapshot, args.event, args.plot, args.plot_dir)


if __name__ == "__main__":
    main()
