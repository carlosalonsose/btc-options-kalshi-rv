"""
2x2 visualization of the Phase 3 + 3.5 findings.

The chart tells the story in four panels:
  A) Kalshi bin liquidity: most bins are not tradable.
  B) yes_bid_size distribution: most bins have little/no bid.
  C) q_buy_repl_disc vs yes_bid_Kalshi for rows with a valid Deribit bracket,
     highlighting the small liquid slice where edge appears.
  D) Net PnL under event-level timing modes (sell-YES only).

Generates data/reports/findings.png from the canonical backtest CSV.

Usage:
  python -m src.runner.visualize_findings                 # canonical run
  python -m src.runner.visualize_findings path/to.csv     # any backtest CSV
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model.fees import kalshi_fee


CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/reports/backtest_canonical.csv")
OUT = Path("data/reports/findings.png")

SPREAD_MAX = 0.03   # liquid-slice spread threshold
SIZE_MIN = 50
EDGE_TH = 0.01      # edge threshold for PnL
DERIBIT_FEE = 0.015


def main() -> None:
    df = pd.read_csv(CSV)
    df["snap_ts"] = pd.to_datetime(df["snap_ts"])
    df = df.dropna(subset=["yes_bid", "yes_ask", "yes_spread",
                           "yes_bid_size", "yes_ask_size", "outcome"]).copy()
    repl = df.dropna(subset=["q_buy_repl_disc"]).copy()

    liq_mask = (
        (df["yes_spread"] <= SPREAD_MAX)
        & (df["yes_bid_size"] >= SIZE_MIN)
        & (df["yes_ask_size"] >= SIZE_MIN)
    )
    repl_liq_mask = (
        (repl["yes_spread"] <= SPREAD_MAX)
        & (repl["yes_bid_size"] >= SIZE_MIN)
        & (repl["yes_ask_size"] >= SIZE_MIN)
    )
    liq = repl[repl_liq_mask].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Cross-Market Relative Value  |  Key Backtest Findings\n"
        f"{len(df):,} rows (bin × snapshot) | {df['event_ticker'].nunique()} events | "
        f"valid discrete Deribit brackets: {len(repl):,} rows",
        fontsize=14, y=0.995,
    )

    # =================== A: histograma yes_spread ===================
    ax = axes[0, 0]
    spread = df["yes_spread"].clip(0, 0.5)
    ax.hist(spread, bins=50, color="tab:gray", alpha=0.85, edgecolor="white")
    ax.axvline(SPREAD_MAX, color="firebrick", lw=2, linestyle="--",
               label=f"slice cutoff = {SPREAD_MAX*100:.0f}c")
    ax.set_xlabel("yes_spread (yes_ask - yes_bid)  [USD]")
    ax.set_ylabel("# bins")
    ax.set_title("A) Kalshi spread: most bins are not tradable")
    ax.legend()
    pct_below = (df["yes_spread"] <= SPREAD_MAX).mean() * 100
    ax.text(0.55, 0.85,
            f"only {pct_below:.0f}% of bins\n"
            f"with spread ≤ 3c",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.grid(True, alpha=0.3)

    # =================== B: yes_bid_size distribution ===================
    ax = axes[0, 1]
    sizes = df["yes_bid_size"].clip(0, 500)
    pct_zero = (df["yes_bid_size"] == 0).mean() * 100
    pct_above = (df["yes_bid_size"] >= SIZE_MIN).mean() * 100
    ax.hist(sizes, bins=50, color="tab:gray", alpha=0.85, edgecolor="white")
    ax.axvline(SIZE_MIN, color="firebrick", lw=2, linestyle="--",
               label=f"slice cutoff = {SIZE_MIN}")
    ax.set_xlabel("yes_bid_size  (capped at 500)")
    ax.set_ylabel("# bins")
    ax.set_title("B) Bid depth: most bins have little/no displayed bid")
    ax.legend()
    ax.text(0.45, 0.6,
            f"{pct_zero:.0f}% with no bid\n"
            f"{pct_above:.0f}% with size ≥ {SIZE_MIN}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.grid(True, alpha=0.3)

    # =================== C: scatter q_buy_repl_disc vs yes_bid ===================
    ax = axes[1, 0]
    nliq = repl[~repl_liq_mask]
    ax.scatter(nliq["q_buy_repl_disc"], nliq["yes_bid"],
               s=4, alpha=0.15, color="lightgray",
               label=f"replicable but illiquid (n={len(nliq):,})")
    win_mask = liq["outcome"] < 0.5
    ax.scatter(liq.loc[win_mask, "q_buy_repl_disc"], liq.loc[win_mask, "yes_bid"],
               s=42, alpha=0.85, color="forestgreen", edgecolors="black", lw=0.4,
               label=f"liquid slice, NO realized (n={int(win_mask.sum())})")
    ax.scatter(liq.loc[~win_mask, "q_buy_repl_disc"], liq.loc[~win_mask, "yes_bid"],
               s=42, alpha=0.85, color="firebrick", edgecolors="black", lw=0.4,
               label=f"liquid slice, YES realized (n={int((~win_mask).sum())})")
    lim = max(0.6, df["yes_bid"].max() * 1.05)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x  (no edge)")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.07, color="green",
                    label="SELL YES edge region")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel("q_buy_repl_disc  =  cost of discrete LONG replication")
    ax.set_ylabel("Kalshi yes_bid  =  price received to sell YES")
    ax.set_title("C) Where edge appears: liquid bins above y=x")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # =================== D: net PnL under per-event aggregation modes ===================
    ax = axes[1, 1]
    colors = {"best": "tab:green", "first": "tab:blue", "random": "tab:orange"}
    summaries = []
    for mode in ["best", "first", "random"]:
        trades = _select_replicated_trades(repl, mode)
        if trades.empty:
            continue
        ax.plot(
            np.arange(1, len(trades) + 1),
            trades["cum_pnl"],
            "o-",
            lw=2,
            markersize=5,
            color=colors[mode],
            label=f"{mode}: {len(trades)} trades, ${trades['pnl'].sum():+.2f}",
        )
        summaries.append((mode, trades))

    ax.axhline(0, color="black", lw=0.7)
    first = dict(summaries).get("first") if summaries else None
    if first is not None:
        ax.text(
            0.05, 0.95,
            "headline assumptions\n"
            "SELL YES only\n"
            f"threshold = {EDGE_TH:.0%}\n"
            f"Deribit fee = ${DERIBIT_FEE:.3f}\n"
            "max 1 trade/event",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow"),
        )
    else:
        ax.text(0.5, 0.5, "(no trades)", ha="center", va="center")

    ax.set_xlabel("# selected event-level trade")
    ax.set_ylabel("Cumulative net PnL  [USD, per contract]")
    ax.set_title("D) Net PnL: discrete replication + fees + timing")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"plot saved: {OUT}")


def _select_replicated_trades(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    d = df.dropna(subset=["yes_bid", "yes_ask", "q_buy_repl_disc", "outcome"]).copy()
    d["sell_edge"] = d["yes_bid"] - d["q_buy_repl_disc"]
    d["sell_score"] = d["sell_edge"] - 0.5 * (d["yes_ask"] - d["yes_bid"]).fillna(0)
    trades = d[d["sell_edge"] > EDGE_TH].copy()

    if mode == "best":
        trades = trades.sort_values("sell_score", ascending=False).drop_duplicates("event_ticker")
    elif mode == "first":
        trades = trades.sort_values("snap_ts").drop_duplicates("event_ticker")
    elif mode == "random":
        trades = trades.sample(frac=1, random_state=42).drop_duplicates("event_ticker")
    else:
        raise ValueError(f"unknown mode: {mode}")

    trades = trades.sort_values("snap_ts").copy()
    trades["fee_total"] = trades["yes_bid"].apply(kalshi_fee) + DERIBIT_FEE
    trades["pnl"] = trades["yes_bid"] - trades["outcome"] - trades["fee_total"]
    trades["cum_pnl"] = trades["pnl"].cumsum()
    return trades


if __name__ == "__main__":
    main()
