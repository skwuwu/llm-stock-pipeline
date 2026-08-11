"""KRX 시세 인제스천.

PER 분자는 **보통주 + 우선주** 시총이어야 한다. KRX 는 우선주를 별도 티커로 상장하므로
(005930 / 005935), 우선주 시총을 모회사에 합산하지 않으면 우선주가 있는 회사가
일괄적으로 저평가로 보인다.

pykrx 가 없으면 CSV 폴백으로 동작한다 — 인제스천 소스를 갈아끼워도 하류가 안 바뀌게
스키마를 이 모듈에서 고정한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

PRICE_COLUMNS = [
    "date", "ticker", "close", "shares_common", "shares_preferred",
    "treasury_shares", "market_cap_common", "market_cap_total", "adtv_20d",
    "price_source", "price_source_disagree", "price_alt_close",
]

# 우선주 티커는 보통주와 앞 5자리를 공유하고 끝자리가 5/6/7/9 인 것이 관례.
# 예외가 존재하므로 최종 판정은 security_master.parent_ticker 를 신뢰한다.
PREFERRED_LAST_DIGITS = {"5", "6", "7", "9"}


def infer_parent_ticker(ticker: str) -> str | None:
    """우선주로 추정되면 보통주 티커를, 아니면 None 을 돌려준다."""
    if len(ticker) != 6 or not ticker.isdigit():
        return None
    if ticker[5] in PREFERRED_LAST_DIGITS and ticker[4] != "0":
        return ticker[:4] + "00"
    if ticker[5] in PREFERRED_LAST_DIGITS:
        return ticker[:5] + "0"
    return None


def aggregate_preferred(raw: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """티커별 원시 시세 → 보통주 기준 행 + 우선주 시총 합산.

    raw   : date, ticker, close, shares, market_cap, adtv_20d
    master: ticker, is_preferred, parent_ticker
    """
    m = master[["ticker", "is_preferred", "parent_ticker"]].copy()
    df = raw.merge(m, on="ticker", how="left")
    df["is_preferred"] = df["is_preferred"].fillna(False).astype(bool)
    df["parent_ticker"] = df["parent_ticker"].where(
        df["parent_ticker"].notna(), df["ticker"].map(infer_parent_ticker))

    common = df[~df["is_preferred"]].copy()
    pref = df[df["is_preferred"]].copy()

    pref_cap = (pref.groupby(["date", "parent_ticker"], as_index=False)
                .agg(pref_cap=("market_cap", "sum"), shares_preferred=("shares", "sum"))
                .rename(columns={"parent_ticker": "ticker"}))

    out = common.merge(pref_cap, on=["date", "ticker"], how="left")
    out["pref_cap"] = out["pref_cap"].fillna(0.0)
    out["shares_preferred"] = out["shares_preferred"].fillna(0).astype("int64")
    out = out.rename(columns={"market_cap": "market_cap_common", "shares": "shares_common"})
    out["market_cap_total"] = out["market_cap_common"] + out["pref_cap"]
    if "treasury_shares" not in out:
        out["treasury_shares"] = 0
    return out.reindex(columns=PRICE_COLUMNS)


def fetch_pykrx(as_of: date, master: pd.DataFrame) -> pd.DataFrame:
    """pykrx 경유 수집. 미설치면 명확히 실패한다(조용히 빈 값 반환 금지)."""
    try:
        from pykrx import stock
    except ImportError as e:
        raise RuntimeError(
            "pykrx 미설치. `pip install pykrx` 또는 --prices-csv 로 CSV 를 넘길 것."
        ) from e

    d = as_of.strftime("%Y%m%d")
    cap = stock.get_market_cap(d).reset_index()
    cap.columns = [c.strip() for c in cap.columns]
    raw = pd.DataFrame({
        "date": as_of,
        "ticker": cap["티커"].astype(str),
        "close": cap["종가"].astype(float),
        "shares": cap["상장주식수"].astype("int64"),
        "market_cap": cap["시가총액"].astype(float),
        "adtv_20d": cap["거래대금"].astype(float),   # 일간값. 20일 평균은 아래에서 대체 가능
    })
    return aggregate_preferred(raw, master)


def load_csv(path: Path, master: pd.DataFrame) -> pd.DataFrame:
    """폴백 경로. 컬럼: date,ticker,close,shares,market_cap,adtv_20d"""
    raw = pd.read_csv(path, dtype={"ticker": str})
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    return aggregate_preferred(raw, master)
