"""
Lectura de snapshots crudos desde disco. Devuelve DataFrames limpios
listos para el modelo. Ningun acceso a red.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def list_snapshots(out_dir: Path | str = "data/snapshots") -> list[Path]:
    out_dir = Path(out_dir)
    return sorted(out_dir.rglob("*.json.gz"))


def load_snapshot(path: Path | str) -> dict:
    with gzip.open(Path(path), "rt", encoding="utf-8") as f:
        return json.load(f)


def latest_snapshot(out_dir: Path | str = "data/snapshots") -> dict:
    paths = list_snapshots(out_dir)
    if not paths:
        raise FileNotFoundError(f"no snapshots in {out_dir}")
    return load_snapshot(paths[-1])


def deribit_btc_usdc_chain(snap: dict) -> pd.DataFrame:
    """DataFrame con todas las opciones BTC_USDC linear vivas en el snapshot.
    Une instrument metadata (strike, expiry, type) con book data (bid/ask/IV)."""
    deribit = snap.get("deribit", {})
    inst = deribit.get("instruments_usdc") or []
    book = deribit.get("book_summary_usdc") or []
    if not isinstance(inst, list) or not isinstance(book, list):
        return pd.DataFrame()

    df_i = pd.DataFrame(inst)
    df_i = df_i[
        (df_i.get("base_currency") == "BTC")
        & (df_i.get("counter_currency") == "USDC")
        & (df_i.get("instrument_type") == "linear")
    ].copy()
    if df_i.empty:
        return pd.DataFrame()

    df_b = pd.DataFrame(book)
    keep_book = [c for c in [
        "instrument_name", "bid_price", "ask_price", "mark_price",
        "bid_iv", "ask_iv", "mark_iv",
        "underlying_price", "underlying_index",
        "open_interest", "volume", "interest_rate",
    ] if c in df_b.columns]
    df_b = df_b[keep_book].copy()

    df = df_i.merge(df_b, on="instrument_name", how="inner")
    df["expiry_dt"] = pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True)
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2.0
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["snapshot_ts_utc"] = pd.to_datetime(snap["snapshot_ts_utc"])
    return df.reset_index(drop=True)


def select_deribit_expiry(chain: pd.DataFrame, after_dt: datetime) -> pd.Timestamp:
    """Devuelve la expiry Deribit MAS PROXIMA pero posterior a after_dt.
    Si no hay ninguna posterior, devuelve la mas cercana en general."""
    after_ts = pd.Timestamp(after_dt).tz_convert("UTC") if pd.Timestamp(after_dt).tzinfo else pd.Timestamp(after_dt, tz="UTC")
    expiries = sorted(chain["expiry_dt"].unique())
    after = [e for e in expiries if e > after_ts]
    if after:
        return pd.Timestamp(after[0])
    return pd.Timestamp(min(expiries, key=lambda e: abs(e - after_ts)))


def kalshi_btc_range(snap: dict, event_ticker: str | None = None) -> pd.DataFrame:
    """DataFrame con bins KXBTC (lower, upper, yes_bid, yes_ask, ...)
    para un event_ticker concreto. Si event_ticker es None, devuelve TODOS los bins
    abiertos del snapshot (puede mezclar varios eventos)."""
    kalshi = snap.get("kalshi", {})
    mkts = kalshi.get("markets_open") or []
    if not isinstance(mkts, list) or not mkts:
        return pd.DataFrame()

    df = pd.DataFrame(mkts)
    if event_ticker is not None:
        df = df[df["event_ticker"] == event_ticker].copy()
        if df.empty:
            return df

    # Schema actual de Kalshi: precios con sufijo _dollars (ya en [0,1]),
    # sizes y volumen con sufijo _fp.
    rename = {
        "yes_bid_dollars": "yes_bid",
        "yes_ask_dollars": "yes_ask",
        "no_bid_dollars": "no_bid",
        "no_ask_dollars": "no_ask",
        "last_price_dollars": "last_price",
        "yes_bid_size_fp": "yes_bid_size",
        "yes_ask_size_fp": "yes_ask_size",
        "volume_fp": "volume",
        "volume_24h_fp": "volume_24h",
        "open_interest_fp": "open_interest",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["lower"] = pd.to_numeric(df.get("floor_strike"), errors="coerce")
    df["upper"] = pd.to_numeric(df.get("cap_strike"), errors="coerce")
    for col in ["yes_bid", "yes_ask", "no_bid", "no_ask", "last_price",
                "yes_bid_size", "yes_ask_size",
                "volume", "volume_24h", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "yes_bid" in df.columns and "yes_ask" in df.columns:
        df["yes_mid"] = (df["yes_bid"] + df["yes_ask"]) / 2.0
        df["yes_spread"] = df["yes_ask"] - df["yes_bid"]

    df["expected_expiration_time"] = pd.to_datetime(
        df.get("expected_expiration_time"), utc=True, errors="coerce"
    )
    df["snapshot_ts_utc"] = pd.to_datetime(snap["snapshot_ts_utc"])
    keep = [c for c in [
        "ticker", "event_ticker", "subtitle", "lower", "upper",
        "yes_bid", "yes_ask", "yes_mid", "yes_spread",
        "yes_bid_size", "yes_ask_size",
        "no_bid", "no_ask", "last_price",
        "volume", "volume_24h", "open_interest",
        "expected_expiration_time", "snapshot_ts_utc",
    ] if c in df.columns]
    return df[keep].sort_values(["lower", "upper"], na_position="last").reset_index(drop=True)


def kalshi_open_events(snap: dict) -> pd.DataFrame:
    """Resumen de eventos KXBTC abiertos en el snapshot, con su settlement."""
    df = kalshi_btc_range(snap)
    if df.empty:
        return df
    return (
        df.groupby("event_ticker", dropna=False)
        .agg(
            n_bins=("ticker", "count"),
            settle=("expected_expiration_time", "first"),
            min_lower=("lower", "min"),
            max_upper=("upper", "max"),
        )
        .reset_index()
        .sort_values("settle")
    )


def select_event_for_settle(snap: dict, target_dt: datetime | None = None) -> str:
    """Elige el event_ticker abierto cuya settlement esta mas cerca de target_dt
    (default: ahora). Util para apuntar al evento mas cercano automaticamente."""
    events = kalshi_open_events(snap)
    if events.empty:
        raise ValueError("snapshot has no open KXBTC events")
    if target_dt is None:
        target_dt = datetime.now(timezone.utc)
    target = pd.Timestamp(target_dt, tz="UTC") if pd.Timestamp(target_dt).tzinfo is None else pd.Timestamp(target_dt).tz_convert("UTC")
    events = events.copy()
    events["dt_to_target"] = (events["settle"] - target).abs()
    return events.sort_values("dt_to_target").iloc[0]["event_ticker"]
