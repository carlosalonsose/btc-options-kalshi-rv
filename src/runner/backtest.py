"""
Backtest offline (Fase 3).

Para cada snapshot grabado:
  1. Corre run_pipeline -> q_deribit por bin del evento mas cercano.
  2. Joinea los bins con el outcome real (Kalshi /markets?status=settled).
  3. Acumula filas (snap_ts, ticker, q_deribit, outcome 0/1, T_K, ...).

Reporta:
  - Brier score y log loss agregados (todos los bins, todos los snapshots).
  - Calibracion por bucket de probabilidad predicha.
  - Reliability por horizonte T_K (cerca vs lejos del settle).
  - Sanity: sum(q_deribit) por snapshot vs 1.

Uso:
  .venv/bin/python -m src.runner.backtest                 # default: stride=5min
  .venv/bin/python -m src.runner.backtest --stride 1      # cada snapshot
  .venv/bin/python -m src.runner.backtest --refresh       # ignora cache settled
  .venv/bin/python -m src.runner.backtest --out data/reports/backtest.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.io.load import list_snapshots, load_snapshot
from src.model.fees import kalshi_fee, total_fee_sell_yes, total_fee_buy_yes
from src.runner.analyze import run_pipeline

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_CACHE = Path("data/kalshi_settled.json.gz")
DEFAULT_OUT = Path("data/reports/backtest.csv")


def _to_utc(dt) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


# -------- fetch settled markets --------------------------------------------

def fetch_kalshi_settled(
    min_close: datetime,
    max_close: datetime,
    cache: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> list[dict]:
    """Pagina /markets?status=settled&series_ticker=KXBTC entre [min_close, max_close]
    y cachea a disco. close_time es cuando el mercado dejo de aceptar trades
    (final del bin horario)."""
    min_close_ts = _to_utc(min_close)
    max_close_ts = _to_utc(max_close)
    if cache.exists() and not refresh:
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            cached = json.load(f)
        cmin = cached.get("min_close")
        cmax = cached.get("max_close")
        if cmin and cmax:
            cmin = pd.Timestamp(cmin)
            cmax = pd.Timestamp(cmax)
            if cmin <= min_close_ts and cmax >= max_close_ts:
                return cached["markets"]

    s = requests.Session()
    s.headers.update({"User-Agent": "cross-market-arb-backtest/0.1"})
    out: list[dict] = []
    cursor = None
    pages = 0
    while True:
        params = {
            "limit": 1000,
            "status": "settled",
            "series_ticker": "KXBTC",
            "min_close_ts": int(min_close_ts.timestamp()),
            "max_close_ts": int(max_close_ts.timestamp()),
        }
        if cursor:
            params["cursor"] = cursor
        r = s.get(f"{KALSHI_BASE}/markets", params=params, timeout=30)
        r.raise_for_status()
        page = r.json()
        markets = page.get("markets") or []
        out.extend(markets)
        pages += 1
        cursor = page.get("cursor")
        if not cursor or not markets:
            break

    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        json.dump({
            "min_close": min_close_ts.isoformat(),
            "max_close": max_close_ts.isoformat(),
            "markets": out,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "pages": pages,
        }, f)
    return out


def settled_lookup(settled: list[dict]) -> dict[str, dict]:
    """ticker -> {result, settlement_value, expected_expiration_time, event_ticker}"""
    out: dict[str, dict] = {}
    for m in settled:
        t = m.get("ticker")
        if not t:
            continue
        out[t] = {
            "result": m.get("result"),
            "settlement_value": m.get("settlement_value"),
            "expected_expiration_time": m.get("expected_expiration_time"),
            "event_ticker": m.get("event_ticker"),
            "floor_strike": m.get("floor_strike"),
            "cap_strike": m.get("cap_strike"),
        }
    return out


# -------- per-snapshot evaluation -----------------------------------------

def evaluate_snapshot(snap_path: Path, lookup: dict[str, dict],
                      min_quality: float = 0.0) -> dict:
    """Corre run_pipeline sobre el snapshot. Devuelve filas (un dict por bin
    cuyo outcome real esta disponible) + diagnosticos."""
    try:
        snap = load_snapshot(snap_path)
        res = run_pipeline(snap)
    except Exception as e:
        return {"snap_path": str(snap_path), "error": str(e), "rows": []}

    # Sprint 2A: filtrar por calidad de expiry.
    quality = res.get("quality")
    quality_score = float(quality.score) if quality is not None else 1.0
    if quality_score < min_quality:
        return {
            "snap_path": str(snap_path),
            "error": f"quality_score={quality_score:.2f} < min={min_quality:.2f}",
            "rows": [],
        }

    annotated = res["annotated"]
    rows: list[dict] = []
    for _, b in annotated.iterrows():
        t = b.get("ticker")
        if not isinstance(t, str):
            continue
        s = lookup.get(t)
        if s is None:
            continue
        result = s.get("result")
        if result not in ("yes", "no"):
            continue
        q = b.get("q_deribit")
        if q is None or not isinstance(q, (int, float)) or not math.isfinite(float(q)):
            continue
        rows.append({
            "snap_ts": res["snap_ts"].isoformat(),
            "event_ticker": res["event_ticker"],
            "ticker": t,
            "lower": _f(b.get("lower")),
            "upper": _f(b.get("upper")),
            "yes_mid": _f(b.get("yes_mid")),
            "yes_bid": _f(b.get("yes_bid")),
            "yes_ask": _f(b.get("yes_ask")),
            "yes_spread": _f(b.get("yes_spread")),
            "yes_bid_size": _f(b.get("yes_bid_size")),
            "yes_ask_size": _f(b.get("yes_ask_size")),
            "volume_24h": _f(b.get("volume_24h")),
            "open_interest": _f(b.get("open_interest")),
            "q_deribit": float(q),
            "q_deribit_p05": _f(b.get("q_deribit_p05")),
            "q_deribit_p95": _f(b.get("q_deribit_p95")),
            "q_deribit_ci_width": _f(b.get("q_deribit_ci_width")),
            "q_buy_exec": _f(b.get("q_buy_exec")),
            "q_sell_exec": _f(b.get("q_sell_exec")),
            "q_buy_repl_disc": _f(b.get("q_buy_repl_disc")),
            "q_sell_repl_disc": _f(b.get("q_sell_repl_disc")),
            "K_low": _f(b.get("K_low")),
            "K_high": _f(b.get("K_high")),
            "outcome": 1 if result == "yes" else 0,
            "T_K": float(res["T_K"]),
            "F": float(res["F"]),
            "expiry_quality": quality_score,
        })
    return {
        "snap_path": str(snap_path),
        "snap_ts": res["snap_ts"].isoformat(),
        "event_ticker": res["event_ticker"],
        "T_K": float(res["T_K"]),
        "sum_q_deribit": float(res["summary"].get("sum_q_deribit") or 0.0),
        "n_bins": int(res["summary"].get("n_bins") or 0),
        "expiry_quality": quality_score,
        "rows": rows,
    }


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


# -------- aggregate metrics ------------------------------------------------

def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_rows": 0}
    q = df["q_deribit"].clip(1e-6, 1 - 1e-6).values
    y = df["outcome"].values
    brier_q = float(np.mean((q - y) ** 2))
    logloss_q = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    out = {
        "n_rows": int(len(df)),
        "n_snapshots": int(df["snap_ts"].nunique()),
        "n_events": int(df["event_ticker"].nunique()),
        "p_yes_realized": float(y.mean()),
        "brier_q_deribit": brier_q,
        "logloss_q_deribit": logloss_q,
    }

    if "yes_mid" in df.columns and df["yes_mid"].notna().any():
        m = df.dropna(subset=["yes_mid"])
        ymid = m["yes_mid"].clip(1e-6, 1 - 1e-6).values
        yo = m["outcome"].values
        out["brier_yes_mid"] = float(np.mean((ymid - yo) ** 2))
        out["logloss_yes_mid"] = float(-np.mean(yo * np.log(ymid) + (1 - yo) * np.log(1 - ymid)))
        out["n_rows_with_mid"] = int(len(m))

    return out


def calibration_table(df: pd.DataFrame, n_bins: int = 10, col: str = "q_deribit") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df = df.copy()
    df["__bucket"] = pd.cut(df[col], bins=edges, include_lowest=True, right=True)
    grp = df.groupby("__bucket", observed=True).agg(
        n=("outcome", "size"),
        mean_pred=(col, "mean"),
        mean_outcome=("outcome", "mean"),
    ).reset_index()
    grp["bias"] = grp["mean_outcome"] - grp["mean_pred"]
    return grp


def executable_pnl(df: pd.DataFrame, threshold: float = 0.01,
                   q_col_buy: str = "q_deribit", q_col_sell: str = "q_deribit",
                   deribit_fee: float = 0.0,
                   one_per_event: bool = False) -> dict:
    """Estrategia simple por-fila (sin sizing, 1 contrato por trade):

      BUY  YES si  q_col_buy[i]  > yes_ask + threshold  -> PnL = outcome - yes_ask
      SELL YES si  yes_bid       > q_col_sell[i] + th   -> PnL = yes_bid - outcome

    Por defecto q_col_buy = q_col_sell = "q_deribit" (mid teorico, Fase 3).
    Para Fase 3.5 pasamos:
        q_col_buy = "q_sell_exec"  (lo que cobramos al short Deribit)
        q_col_sell = "q_buy_exec"  (lo que pagamos al long Deribit)

    Fees (Sprint 1, opcional, por defecto 0):
        kalshi_fee(price) aplicado por trade segun el precio del fill.
        deribit_fee constante anadido por trade.

    one_per_event: si True, agrupar por event_ticker y quedarse SOLO con el
        trade de mejor "score" (= edge - 0.5*spread) en cada lado. Esto
        descorrelaciona observaciones del mismo evento.
    """
    if df.empty or "yes_ask" not in df.columns or "yes_bid" not in df.columns:
        return {"n_trades": 0}
    needed = ["yes_ask", "yes_bid", q_col_buy, q_col_sell, "outcome"]
    d = df.dropna(subset=needed).copy()
    if d.empty:
        return {"n_trades": 0}

    d["__buy_edge"] = d[q_col_buy] - d["yes_ask"]
    d["__sell_edge"] = d["yes_bid"] - d[q_col_sell]
    spread = (d["yes_ask"] - d["yes_bid"]).fillna(0)
    d["__buy_score"] = d["__buy_edge"] - 0.5 * spread
    d["__sell_score"] = d["__sell_edge"] - 0.5 * spread

    buy_mask = d["__buy_edge"] > threshold
    sell_mask = d["__sell_edge"] > threshold

    if one_per_event and "event_ticker" in d.columns:
        if buy_mask.any():
            best_buy_idx = (d[buy_mask].sort_values("__buy_score", ascending=False)
                                       .drop_duplicates("event_ticker").index)
            buy_mask = d.index.isin(best_buy_idx)
        if sell_mask.any():
            best_sell_idx = (d[sell_mask].sort_values("__sell_score", ascending=False)
                                         .drop_duplicates("event_ticker").index)
            sell_mask = d.index.isin(best_sell_idx)

    fee_kalshi_buy = d.loc[buy_mask, "yes_ask"].apply(kalshi_fee)
    fee_kalshi_sell = d.loc[sell_mask, "yes_bid"].apply(kalshi_fee)

    buy_pnl_gross = (d.loc[buy_mask, "outcome"] - d.loc[buy_mask, "yes_ask"]).sum()
    sell_pnl_gross = (d.loc[sell_mask, "yes_bid"] - d.loc[sell_mask, "outcome"]).sum()
    buy_fees = float(fee_kalshi_buy.sum() + deribit_fee * int(buy_mask.sum()))
    sell_fees = float(fee_kalshi_sell.sum() + deribit_fee * int(sell_mask.sum()))
    buy_pnl = float(buy_pnl_gross) - buy_fees
    sell_pnl = float(sell_pnl_gross) - sell_fees

    out = {
        "threshold": threshold,
        "n_buy_yes": int(buy_mask.sum()),
        "n_sell_yes": int(sell_mask.sum()),
        "buy_pnl_gross": float(buy_pnl_gross),
        "sell_pnl_gross": float(sell_pnl_gross),
        "buy_fees": buy_fees,
        "sell_fees": sell_fees,
        "buy_pnl_total": float(buy_pnl),
        "sell_pnl_total": float(sell_pnl),
        "total_pnl": float(buy_pnl + sell_pnl),
    }
    if int(buy_mask.sum()) > 0:
        out["buy_pnl_avg_per_trade"] = float(buy_pnl / buy_mask.sum())
        out["buy_winrate"] = float((d.loc[buy_mask, "outcome"] >= 0.5).mean())
    if int(sell_mask.sum()) > 0:
        out["sell_pnl_avg_per_trade"] = float(sell_pnl / sell_mask.sum())
        out["sell_winrate"] = float((d.loc[sell_mask, "outcome"] < 0.5).mean())
    return out


def liquidity_grid(df: pd.DataFrame,
                   threshold: float = 0.01,
                   max_spread_levels=(1.0, 0.10, 0.05, 0.03, 0.02, 0.01),
                   min_size_levels=(0, 10, 50, 100, 500),
                   q_col_buy: str = "q_sell_exec",
                   q_col_sell: str = "q_buy_exec",
                   deribit_fee: float = 0.0,
                   one_per_event: bool = False) -> pd.DataFrame:
    """Para cada combinacion (max_spread, min_size), aplica filtro y calcula PnL
    con un threshold fijo. Por defecto usa Q ejecutable (Fase 3.5)."""
    if df.empty or "yes_spread" not in df.columns:
        return pd.DataFrame()
    base = df.dropna(subset=["yes_spread", "yes_bid_size", "yes_ask_size",
                             q_col_buy, q_col_sell, "outcome"]).copy()
    if base.empty:
        return pd.DataFrame()

    grid = []
    for ms in max_spread_levels:
        for mn in min_size_levels:
            f = base[
                (base["yes_spread"] <= ms)
                & (base["yes_bid_size"] >= mn)
                & (base["yes_ask_size"] >= mn)
            ]
            if f.empty:
                grid.append({"max_spread": ms, "min_size": mn,
                             "n_rows": 0, "n_buy": 0, "n_sell": 0,
                             "buy_pnl": 0.0, "sell_pnl": 0.0, "total": 0.0,
                             "buy_avg": float("nan"), "sell_avg": float("nan")})
                continue
            p = executable_pnl(f, threshold=threshold,
                               q_col_buy=q_col_buy, q_col_sell=q_col_sell,
                               deribit_fee=deribit_fee,
                               one_per_event=one_per_event)
            grid.append({
                "max_spread": ms,
                "min_size": mn,
                "n_rows": int(len(f)),
                "n_buy": int(p.get("n_buy_yes", 0)),
                "n_sell": int(p.get("n_sell_yes", 0)),
                "buy_pnl": float(p.get("buy_pnl_total", 0.0)),
                "sell_pnl": float(p.get("sell_pnl_total", 0.0)),
                "total": float(p.get("total_pnl", 0.0)),
                "buy_avg": p.get("buy_pnl_avg_per_trade", float("nan")),
                "sell_avg": p.get("sell_pnl_avg_per_trade", float("nan")),
            })
    return pd.DataFrame(grid)


def horizon_table(df: pd.DataFrame, edges_minutes=(0, 5, 15, 30, 60)) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    YEAR_MIN = 365.0 * 24.0 * 60.0
    df = df.copy()
    df["T_K_min"] = df["T_K"] * YEAR_MIN
    edges = list(edges_minutes) + [float("inf")]
    df["__h"] = pd.cut(df["T_K_min"], bins=edges, right=False)
    grp = df.groupby("__h", observed=True)[["outcome", "q_deribit"]].apply(
        lambda g: pd.Series({
            "n": len(g),
            "p_yes": g["outcome"].mean(),
            "brier_q": ((g["q_deribit"].clip(1e-6, 1-1e-6) - g["outcome"]) ** 2).mean(),
        })
    ).reset_index()
    return grp


# -------- driver -----------------------------------------------------------

def run(
    snapshots_dir: Path,
    out_csv: Path,
    stride: int,
    refresh_cache: bool,
    limit: int | None,
    deribit_fee: float = 0.015,
    one_per_event: bool = True,
    min_quality: float = 0.0,
) -> None:
    paths = list_snapshots(snapshots_dir)
    if not paths:
        print(f"no snapshots in {snapshots_dir}")
        return
    if stride > 1:
        paths = paths[::stride]
    if limit:
        paths = paths[:limit]

    # Rango temporal a fetchear de Kalshi: cubrimos snapshot ts +- 1 dia.
    first = pd.Timestamp(load_snapshot(paths[0])["snapshot_ts_utc"])
    last = pd.Timestamp(load_snapshot(paths[-1])["snapshot_ts_utc"])
    if first.tzinfo is None:
        first = first.tz_localize("UTC")
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    min_close = first - timedelta(hours=2)
    max_close = last + timedelta(hours=2)

    print(f"fetching Kalshi settled markets between {min_close} and {max_close}...")
    settled = fetch_kalshi_settled(min_close, max_close, refresh=refresh_cache)
    lookup = settled_lookup(settled)
    print(f"  cached settled markets: {len(settled)} ({len(lookup)} unique tickers)")

    diagnostics: list[dict] = []
    all_rows: list[dict] = []
    t0 = time.time()
    n = len(paths)
    print(f"evaluating {n} snapshots (stride={stride}, min_quality={min_quality:.2f})...")
    for i, p in enumerate(paths, 1):
        out = evaluate_snapshot(p, lookup, min_quality=min_quality)
        if "error" in out and out["error"]:
            diagnostics.append({"snap_path": out["snap_path"], "error": out["error"],
                                 "expiry_quality": out.get("expiry_quality", float("nan"))})
        else:
            diagnostics.append({
                "snap_path": out["snap_path"],
                "snap_ts": out["snap_ts"],
                "event_ticker": out["event_ticker"],
                "T_K": out["T_K"],
                "sum_q_deribit": out["sum_q_deribit"],
                "n_bins": out["n_bins"],
                "n_matched": len(out["rows"]),
                "expiry_quality": out.get("expiry_quality", float("nan")),
            })
        all_rows.extend(out.get("rows", []))
        if i % 10 == 0 or i == n:
            dt = time.time() - t0
            print(f"  [{i}/{n}] {p.name}  rows_so_far={len(all_rows)}  elapsed={dt:.1f}s")

    df = pd.DataFrame(all_rows)
    diag_df = pd.DataFrame(diagnostics)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    diag_csv = out_csv.with_name(out_csv.stem + "_diag.csv")
    diag_df.to_csv(diag_csv, index=False)
    print(f"\nrows guardadas: {out_csv}  ({len(df)} filas)")
    print(f"diag guardado:  {diag_csv}  ({len(diag_df)} snapshots)")

    print("\n" + "=" * 70)
    print("  MÉTRICAS AGREGADAS")
    print("=" * 70)
    m = metrics(df)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:25s} = {v:.6f}")
        else:
            print(f"  {k:25s} = {v}")

    print("\n" + "=" * 70)
    print("  CALIBRACIÓN q_deribit (10 buckets)")
    print("=" * 70)
    cal = calibration_table(df, n_bins=10, col="q_deribit")
    if not cal.empty:
        print(cal.to_string(index=False))

    if "yes_mid" in df.columns and df["yes_mid"].notna().any():
        print("\n" + "=" * 70)
        print("  CALIBRACIÓN yes_mid Kalshi (10 buckets)")
        print("=" * 70)
        cal_k = calibration_table(df.dropna(subset=["yes_mid"]), n_bins=10, col="yes_mid")
        print(cal_k.to_string(index=False))

    print("\n" + "=" * 70)
    print("  RELIABILITY POR HORIZONTE T_K")
    print("=" * 70)
    hor = horizon_table(df)
    if not hor.empty:
        print(hor.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"  PnL ejecutable (fase 3.5) — sweep thresholds  |  fees activas")
    print(f"  fee_kalshi=ceil(0.07*p*(1-p)) USD,  fee_deribit_const={deribit_fee:.3f} USD")
    print(f"  one_per_event={one_per_event}")
    print("=" * 70)
    has_exec = "q_buy_exec" in df.columns and df["q_buy_exec"].notna().any()
    rows_gross: list[dict] = []
    rows_net: list[dict] = []
    for th in [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]:
        if has_exec:
            p_gross = executable_pnl(df, threshold=th,
                                     q_col_buy="q_sell_exec",
                                     q_col_sell="q_buy_exec",
                                     deribit_fee=0.0,
                                     one_per_event=one_per_event)
            p_net = executable_pnl(df, threshold=th,
                                   q_col_buy="q_sell_exec",
                                   q_col_sell="q_buy_exec",
                                   deribit_fee=deribit_fee,
                                   one_per_event=one_per_event)
            rows_gross.append({
                "th": th,
                "n_buy": p_gross.get("n_buy_yes", 0),
                "n_sell": p_gross.get("n_sell_yes", 0),
                "buy_pnl": p_gross.get("buy_pnl_total", 0.0),
                "sell_pnl": p_gross.get("sell_pnl_total", 0.0),
                "total": p_gross.get("total_pnl", 0.0),
            })
            rows_net.append({
                "th": th,
                "n_buy": p_net.get("n_buy_yes", 0),
                "n_sell": p_net.get("n_sell_yes", 0),
                "fees_total": p_net.get("buy_fees", 0.0) + p_net.get("sell_fees", 0.0),
                "buy_pnl": p_net.get("buy_pnl_total", 0.0),
                "sell_pnl": p_net.get("sell_pnl_total", 0.0),
                "total": p_net.get("total_pnl", 0.0),
            })

    print("\n  [GROSS] sin descontar fees:")
    print(pd.DataFrame(rows_gross).to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
    print("\n  [NET] descontando fees Kalshi + Deribit:")
    print(pd.DataFrame(rows_net).to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))

    # Sprint 1.5: PnL contra replicacion discreta REAL con strikes Deribit.
    has_repl = "q_buy_repl_disc" in df.columns and df["q_buy_repl_disc"].notna().any()
    if has_repl:
        n_repl = int(df["q_buy_repl_disc"].notna().sum())
        print("\n" + "=" * 70)
        print(f"  PnL contra REPLICACION DISCRETA (strikes Deribit reales)")
        print(f"  Filas con replicacion ejecutable: {n_repl} / {len(df):,}")
        print(f"  (resto: bin fuera del bracket Deribit o quotes invalidos)")
        print("=" * 70)
        rows_repl: list[dict] = []
        for th in [0.005, 0.01, 0.02, 0.03, 0.05]:
            p = executable_pnl(df, threshold=th,
                               q_col_buy="q_sell_repl_disc",
                               q_col_sell="q_buy_repl_disc",
                               deribit_fee=deribit_fee,
                               one_per_event=one_per_event)
            rows_repl.append({
                "th": th,
                "n_buy": p.get("n_buy_yes", 0),
                "n_sell": p.get("n_sell_yes", 0),
                "fees": p.get("buy_fees", 0.0) + p.get("sell_fees", 0.0),
                "buy_pnl": p.get("buy_pnl_total", 0.0),
                "sell_pnl": p.get("sell_pnl_total", 0.0),
                "total": p.get("total_pnl", 0.0),
            })
        print(pd.DataFrame(rows_repl).to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
        print("  nota: NaN bins descartados (no hay bracket Deribit ejecutable)")

    if has_exec and "yes_spread" in df.columns:
        print("\n" + "=" * 70)
        print("  GRID DE LIQUIDEZ — PnL NETO exec con threshold=1% por (spread, size)")
        print("=" * 70)
        print("  (filas filtradas: yes_spread <= max_spread Y bid/ask sizes >= min_size)")
        grid = liquidity_grid(df, threshold=0.01,
                              deribit_fee=deribit_fee,
                              one_per_event=one_per_event)
        if grid.empty:
            print("  (sin datos)")
        else:
            print(grid.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))
            best = grid.sort_values("total", ascending=False).head(3)
            print("\n  TOP-3 combinaciones por total_pnl:")
            print(best.to_string(index=False, float_format=lambda x: f"{x:>9.4f}"))

    print("\n" + "=" * 70)
    print("  CALIDAD DE EXPIRY DERIBIT (Sprint 2)")
    print("=" * 70)
    if not diag_df.empty and "expiry_quality" in diag_df.columns:
        q_scores = diag_df["expiry_quality"].dropna()
        if not q_scores.empty:
            print(f"  snapshots totales procesados : {len(diag_df)}")
            print(f"  con quality >= 0.25 (usable) : {int((q_scores >= 0.25).sum())}")
            print(f"  con quality >= 0.50          : {int((q_scores >= 0.50).sum())}")
            print(f"  con quality >= 0.75 (alta)   : {int((q_scores >= 0.75).sum())}")
            print(f"  median quality               : {q_scores.median():.3f}")
            print(f"  p25 / p75                    : {q_scores.quantile(0.25):.3f} / {q_scores.quantile(0.75):.3f}")
    if "expiry_quality" in df.columns:
        q_row = df["expiry_quality"].dropna()
        if not q_row.empty:
            print(f"  quality en filas con outcomes: media={q_row.mean():.3f}  median={q_row.median():.3f}")

    print("\n" + "=" * 70)
    print("  SANITY: sum_q_deribit distribución (debe ≈ 1)")
    print("=" * 70)
    if not diag_df.empty and "sum_q_deribit" in diag_df.columns:
        s = diag_df["sum_q_deribit"].dropna()
        if not s.empty:
            print(f"  median={s.median():.4f}  mean={s.mean():.4f}")
            print(f"  min={s.min():.4f}  max={s.max():.4f}")
            mask = (s - 1).abs() > 0.01
            far = diag_df.loc[mask.index[mask]]
            if not far.empty:
                print(f"  WARN: {len(far)} snapshots con |sum_q - 1| > 0.01")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Backtest offline del pipeline arb")
    p.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stride", type=int, default=5,
                   help="usa 1 cada N snapshots (default 5 = ~5 min)")
    p.add_argument("--refresh", action="store_true",
                   help="re-fetch Kalshi settled (ignora cache)")
    p.add_argument("--limit", type=int, default=None,
                   help="limita a los primeros N snapshots (para smoke test)")
    p.add_argument("--fees-deribit", type=float, default=0.015,
                   help="fee constante por contrato Deribit en USD (default 0.015)")
    p.add_argument("--all-trades", action="store_true",
                   help="contar TODOS los trades (default: max 1 por evento)")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="descartar snapshots con expiry_quality < umbral (0-1, default 0 = no filtro)")
    args = p.parse_args(argv)
    run(args.snapshots, args.out, args.stride, args.refresh, args.limit,
        deribit_fee=args.fees_deribit,
        one_per_event=not args.all_trades,
        min_quality=args.min_quality)


if __name__ == "__main__":
    main()
