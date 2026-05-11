"""
Visualizacion 2x2 de los hallazgos Fase 3 + 3.5.

Cuenta la historia en 4 paneles:
  A) Liquidez de bins Kalshi: la mayoria son ilíquidos.
  B) yes_bid_size distribution: la mayoria no tienen bid.
  C) Scatter q_buy_exec_Deribit vs yes_bid_Kalshi en TODO el universo,
     resaltando los pocos bins ultra-liquidos donde aparece edge.
  D) PnL acumulado realizado en el slice ultra-liquido (sell-yes only).

Genera data/reports/findings.png. No tiene CLI: solo importa pandas + matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV = Path("data/reports/backtest.csv")
OUT = Path("data/reports/findings.png")

SPREAD_MAX = 0.03   # umbral del slice ultra-liquido
SIZE_MIN = 50
EDGE_TH = 0.01      # threshold de edge para PnL


def main() -> None:
    df = pd.read_csv(CSV)
    df["snap_ts"] = pd.to_datetime(df["snap_ts"])
    df = df.dropna(subset=["yes_bid", "yes_ask", "yes_spread",
                            "yes_bid_size", "yes_ask_size",
                            "q_buy_exec", "q_sell_exec", "outcome"]).copy()

    liq_mask = (
        (df["yes_spread"] <= SPREAD_MAX)
        & (df["yes_bid_size"] >= SIZE_MIN)
        & (df["yes_ask_size"] >= SIZE_MIN)
    )
    liq = df[liq_mask].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        "Cross-Market Arbitrage  |  hallazgos Fase 3.5\n"
        f"{len(df):,} filas (bin × snapshot) | {df['event_ticker'].nunique()} eventos | "
        f"slice liquido: {len(liq)} filas",
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
    ax.set_title("A) Spread Kalshi: la mayoría de bins NO son tradables")
    ax.legend()
    pct_below = (df["yes_spread"] <= SPREAD_MAX).mean() * 100
    ax.text(0.55, 0.85,
            f"solo {pct_below:.0f}% de bins\n"
            f"con spread ≤ 3c",
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
    ax.set_title("B) Profundidad del bid: la mayoría de bins NO tienen bid")
    ax.legend()
    ax.text(0.45, 0.6,
            f"{pct_zero:.0f}% sin bid\n"
            f"{pct_above:.0f}% con size ≥ {SIZE_MIN}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))
    ax.grid(True, alpha=0.3)

    # =================== C: scatter q_buy_exec vs yes_bid ===================
    ax = axes[1, 0]
    nliq = df[~liq_mask]
    ax.scatter(nliq["q_buy_exec"], nliq["yes_bid"],
               s=4, alpha=0.15, color="lightgray",
               label=f"resto de bins (n={len(nliq):,})")
    win_mask = liq["outcome"] < 0.5
    ax.scatter(liq.loc[win_mask, "q_buy_exec"], liq.loc[win_mask, "yes_bid"],
               s=42, alpha=0.85, color="forestgreen", edgecolors="black", lw=0.4,
               label=f"slice líquido, NO realizado (n={int(win_mask.sum())})")
    ax.scatter(liq.loc[~win_mask, "q_buy_exec"], liq.loc[~win_mask, "yes_bid"],
               s=42, alpha=0.85, color="firebrick", edgecolors="black", lw=0.4,
               label=f"slice líquido, YES realizado (n={int((~win_mask).sum())})")
    lim = max(0.6, df["yes_bid"].max() * 1.05)
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x  (sin edge)")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.07, color="green",
                    label="zona SELL YES con edge")
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_xlabel("q_buy_exec  =  costo replicación LONG en Deribit")
    ax.set_ylabel("yes_bid Kalshi  =  precio al que vendes YES")
    ax.set_title("C) Donde aparece el edge: bins líquidos por encima de y=x")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # =================== D: PnL acumulado en slice ===================
    ax = axes[1, 1]
    sell_mask = (liq["yes_bid"] - liq["q_buy_exec"]) > EDGE_TH
    trades = liq[sell_mask].sort_values("snap_ts").copy()
    trades["pnl"] = trades["yes_bid"] - trades["outcome"]
    trades["cum_pnl"] = trades["pnl"].cumsum()
    trades["t_idx"] = np.arange(1, len(trades) + 1)

    if len(trades) > 0:
        ax.plot(trades["t_idx"], trades["cum_pnl"], "o-",
                color="forestgreen", lw=2, markersize=6)
        ax.axhline(0, color="black", lw=0.7)
        for ev, sub in trades.groupby("event_ticker"):
            ax.axvline(sub["t_idx"].iloc[0], color="gray", alpha=0.15, lw=1)
        final = trades["cum_pnl"].iloc[-1]
        avg = trades["pnl"].mean()
        wr = (trades["outcome"] < 0.5).mean()
        ax.text(0.05, 0.95,
                f"n_trades = {len(trades)}\n"
                f"PnL final = ${final:+.2f}\n"
                f"avg/trade = ${avg:+.4f}\n"
                f"winrate   = {wr:.1%}\n"
                f"events    = {trades['event_ticker'].nunique()}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round", facecolor="lightyellow"))
    else:
        ax.text(0.5, 0.5, "(sin trades)", ha="center", va="center")

    ax.set_xlabel("# trade (orden cronológico)")
    ax.set_ylabel("PnL acumulado  [USD, por contrato]")
    ax.set_title(f"D) Realizado: SELL YES en slice líquido, threshold={EDGE_TH*100:.0f}%")
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"plot guardado: {OUT}")


if __name__ == "__main__":
    main()
