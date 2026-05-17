"""
Passive-fill backtest with an adverse-selection model.

The taker backtest (`backtest.py`) crosses the Kalshi spread and loses. The
only economically interesting question left is whether *posting* a tighter
two-sided market around the Deribit fair value survives adverse selection.

Strategy modelled (maker):
  At snapshot t, fair = q_deribit_t. Quote a market tighter than Kalshi's:
      our_ask = fair + h   (offer to SELL YES)
      our_bid = fair - h   (bid to BUY YES)
  A side is posted only if it strictly improves the Kalshi book and the side
  is hedgeable with a discrete Deribit vertical.

Passive fill (this is the whole point):
  A resting quote does not fill at t. It fills only if the market trades
  *through* it by the next snapshot t+1 for the same ticker:
      SELL YES @ our_ask  fills if  yes_bid_{t+1} >= our_ask
      BUY  YES @ our_bid  fills if  yes_ask_{t+1} <= our_bid
  On fill you hedge at the *contemporaneous* (t+1) executable replication
  price. Adverse selection is encoded automatically: you only get filled
  when the market (and BTC) moved, and the hedge is repriced by that move.

Three measurements are reported so the cost of adverse selection is explicit:
  - naive   : fill + hedge evaluated at t (no staleness). Optimistic control.
  - adverse : fill + hedge evaluated at t+1. The honest number.
  - delta   : adverse - naive (what staleness costs you).

Usage:
  python -m src.runner.passive_fill                       # canonical csv
  python -m src.runner.passive_fill data/reports/backtest_sample.csv
  python -m src.runner.passive_fill --half-spread 0.01 --max-tk-min 15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.fees import kalshi_fee

YEAR_MIN = 365.0 * 24.0 * 60.0
DEFAULT_CSV = Path("data/reports/backtest_canonical.csv")


def _fill_side(df: pd.DataFrame, h: float, deribit_fee: float,
               adverse: bool) -> pd.DataFrame:
    """One row per (ticker) consecutive snapshot pair that produced a fill.

    adverse=True : a posted quote fills only if the market trades through it
                   by t+1; hedge priced at t+1 (the honest model).
    adverse=False: every posted quote is assumed to fill at its price, hedged
                   at t. This is an optimistic upper bound with no adverse
                   selection and no fill uncertainty, used only as a control.
    """
    d = df.sort_values(["ticker", "snap_ts"]).copy()
    g = d.groupby("ticker", sort=False)
    # next-snapshot view of the same ticker
    nxt = g.shift(-1)
    d["yes_bid_n"] = nxt["yes_bid"]
    d["yes_ask_n"] = nxt["yes_ask"]
    d["q_buy_repl_n"] = nxt["q_buy_repl_disc"]
    d["q_sell_repl_n"] = nxt["q_sell_repl_disc"]
    d = d[nxt["snap_ts"].notna()].copy()  # drop last snapshot per ticker

    fair = d["q_deribit"]
    our_ask = fair + h
    our_bid = fair - h

    # hedge price: contemporaneous with the fill (t+1 if adverse, else t)
    hedge_buy = d["q_buy_repl_n"] if adverse else d["q_buy_repl_disc"]
    hedge_sell = d["q_sell_repl_n"] if adverse else d["q_sell_repl_disc"]

    # post a side only if it strictly improves the Kalshi book, does not cross,
    # and is hedgeable with a discrete Deribit vertical
    post_sell = (our_ask < d["yes_ask"]) & (our_ask > d["yes_bid"]) & hedge_buy.notna()
    post_buy = (our_bid > d["yes_bid"]) & (our_bid < d["yes_ask"]) & hedge_sell.notna()

    if adverse:
        # resting quote fills only if the market trades through it by t+1
        fill_sell = post_sell & (d["yes_bid_n"] >= our_ask)
        fill_buy = post_buy & (d["yes_ask_n"] <= our_bid)
    else:
        # optimistic control: every posted quote is assumed to fill
        fill_sell = post_sell
        fill_buy = post_buy

    rows = []
    s = d[fill_sell]
    if not s.empty:
        a = our_ask[fill_sell]
        pnl_hedged = a - s["q_buy_repl_n" if adverse else "q_buy_repl_disc"] \
            - a.map(kalshi_fee) - deribit_fee
        pnl_unhedged = a - s["outcome"] - a.map(kalshi_fee)
        rows.append(pd.DataFrame({
            "ticker": s["ticker"], "event_ticker": s["event_ticker"],
            "snap_ts": s["snap_ts"], "side": "sell",
            "price": a, "T_K_min": s["T_K"] * YEAR_MIN,
            "pnl_hedged": pnl_hedged, "pnl_unhedged": pnl_unhedged,
            "outcome": s["outcome"],
        }))
    b = d[fill_buy]
    if not b.empty:
        p = our_bid[fill_buy]
        pnl_hedged = b["q_sell_repl_n" if adverse else "q_sell_repl_disc"] - p \
            - p.map(kalshi_fee) - deribit_fee
        pnl_unhedged = b["outcome"] - p - p.map(kalshi_fee)
        rows.append(pd.DataFrame({
            "ticker": b["ticker"], "event_ticker": b["event_ticker"],
            "snap_ts": b["snap_ts"], "side": "buy",
            "price": p, "T_K_min": b["T_K"] * YEAR_MIN,
            "pnl_hedged": pnl_hedged, "pnl_unhedged": pnl_unhedged,
            "outcome": b["outcome"],
        }))
    if not rows:
        return pd.DataFrame(columns=["ticker", "event_ticker", "snap_ts",
                                     "side", "price", "T_K_min",
                                     "pnl_hedged", "pnl_unhedged", "outcome"])
    return pd.concat(rows, ignore_index=True)


def _dedup_per_event(trades: pd.DataFrame, one_per_event: bool) -> pd.DataFrame:
    if not one_per_event or trades.empty:
        return trades
    return (trades.sort_values("snap_ts")
                  .drop_duplicates(["event_ticker", "side"]))


def _summary(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"variant": label, "n": 0}
    return {
        "variant": label,
        "n": int(len(trades)),
        "n_events": int(trades["event_ticker"].nunique()),
        "pnl_hedged": float(trades["pnl_hedged"].sum()),
        "pnl_unhedged": float(trades["pnl_unhedged"].sum()),
        "avg_hedged": float(trades["pnl_hedged"].mean()),
        "winrate_hedged": float((trades["pnl_hedged"] > 0).mean()),
    }


def run(csv: Path, half_spread: float, max_tk_min: float | None,
        max_spread: float, min_size: float, deribit_fee: float,
        one_per_event: bool) -> None:
    df = pd.read_csv(csv)
    df["snap_ts"] = pd.to_datetime(df["snap_ts"])
    base = df.dropna(subset=["q_deribit", "yes_bid", "yes_ask",
                             "yes_spread", "outcome", "T_K"]).copy()

    # restrictions: liquid, near-settle, hedgeable handled inside _fill_side
    base = base[(base["yes_spread"] <= max_spread)
                & (base["yes_bid_size"].fillna(0) >= min_size)
                & (base["yes_ask_size"].fillna(0) >= min_size)]
    if max_tk_min is not None:
        base = base[base["T_K"] * YEAR_MIN <= max_tk_min]

    print("=" * 72)
    print("  PASSIVE-FILL BACKTEST WITH ADVERSE-SELECTION MODEL")
    print("=" * 72)
    print(f"  csv             = {csv}")
    print(f"  half_spread h   = {half_spread:.4f}  (quote = q_deribit ± h)")
    print(f"  near-settle     = T_K <= {max_tk_min} min"
          if max_tk_min is not None else "  near-settle     = (no filter)")
    print(f"  liquidity       = spread <= {max_spread}, size >= {min_size}")
    print(f"  deribit_fee     = {deribit_fee:.3f}   one_per_event = {one_per_event}")
    print(f"  rows after restrictions = {len(base):,}")
    print("-" * 72)

    out = []
    for adverse, label in [(False, "naive (fill@t, no staleness)"),
                           (True, "adverse (fill@t+1, honest)")]:
        tr = _dedup_per_event(_fill_side(base, half_spread, deribit_fee, adverse),
                              one_per_event)
        out.append(_summary(tr, label))
    res = pd.DataFrame(out)
    print(res.to_string(index=False, float_format=lambda x: f"{x:>10.4f}"))

    if len(res) == 2 and res.iloc[0]["n"] and res.iloc[1]["n"]:
        d = res.iloc[1]["pnl_hedged"] - res.iloc[0]["pnl_hedged"]
        print("-" * 72)
        print(f"  ADVERSE-SELECTION COST (hedged): "
              f"{d:+.4f} USD over the run "
              f"({res.iloc[1]['n']} vs {res.iloc[0]['n']} fills)")
        print("  If 'adverse' is <= 0 the maker idea does not survive in this"
              " data.")

    print("-" * 72)
    print("  half-spread sweep (adverse, hedged):")
    sweep = []
    for h in [0.005, 0.01, 0.02, 0.03, 0.05]:
        tr = _dedup_per_event(_fill_side(base, h, deribit_fee, True),
                              one_per_event)
        sm = _summary(tr, f"h={h}")
        sweep.append({"h": h, "n": sm.get("n", 0),
                      "n_events": sm.get("n_events", 0),
                      "pnl_hedged": sm.get("pnl_hedged", 0.0),
                      "avg": sm.get("avg_hedged", float("nan"))})
    print(pd.DataFrame(sweep).to_string(index=False,
                                        float_format=lambda x: f"{x:>10.4f}"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Passive-fill / adverse-selection backtest")
    p.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    p.add_argument("--half-spread", type=float, default=0.01)
    p.add_argument("--max-tk-min", type=float, default=15.0,
                   help="solo bins a <= N min del settle (None = sin filtro)")
    p.add_argument("--max-spread", type=float, default=0.05)
    p.add_argument("--min-size", type=float, default=50.0)
    p.add_argument("--fees-deribit", type=float, default=0.015)
    p.add_argument("--all-trades", action="store_true",
                   help="no de-correlar por evento")
    a = p.parse_args(argv)
    run(a.csv, a.half_spread,
        None if a.max_tk_min < 0 else a.max_tk_min,
        a.max_spread, a.min_size, a.fees_deribit,
        one_per_event=not a.all_trades)


if __name__ == "__main__":
    main()
