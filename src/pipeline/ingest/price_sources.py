"""가격 소스 추상화 + 소스 간 교차검증.

단일 소스로는 주가 오류를 잡을 수 없다. 시총·PER·PBR 이 전부 같은 종가에서
나오므로 종가가 틀리면 셋이 함께 틀리고 서로를 검증하지 못한다.

실측(2026-08-07): FDR 이 26일치 정상 시계열로 대한제분 11,510원을 주는데
DART 배당은 153,846원을 함의한다. 액면분할도 아니다(1Q2026 과 FY2025 주식수 동일).
**어느 쪽이 맞는지 판정할 세 번째 근거가 없어** 지금은 배제로 처리하고 있다.
두 번째 가격 소스가 붙으면 그 11건이 '배제'에서 '확정'으로 바뀐다.

규율:
  - 소스가 어긋나면 값을 지우지 않고 **플래그를 단다.** 어느 쪽이 맞는지 모를 때
    임의로 하나를 고르면, 틀린 값을 확신을 갖고 쓰는 셈이 된다.
  - 검증되지 않은 소스는 **프로브를 통과하기 전까지 대량 수집에 쓸 수 없다.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

# 소스 간 종가가 이보다 벌어지면 불일치로 본다. 같은 거래일 종가는 원래 같아야
# 하므로 좁게 잡는다 — 넓게 두면 교차검증의 의미가 없다.
PRICE_AGREEMENT_TOLERANCE = 0.02
OUT_COLUMNS = ["ticker", "close", "adtv_20d", "price_date"]


class PriceSourceError(RuntimeError):
    pass


class PriceSource(Protocol):
    name: str

    def fetch(self, tickers: list[str], as_of: date,
              refresh: bool = False) -> pd.DataFrame:
        """ticker, close, adtv_20d, price_date 컬럼을 가진 프레임."""
        ...


@dataclass
class FdrSource:
    """FinanceDataReader. 현재 운영 소스 — PriceIngestor 를 감싼다."""

    cache_root: Path
    name: str = "fdr"

    def fetch(self, tickers: list[str], as_of: date,
              refresh: bool = False) -> pd.DataFrame:
        from pipeline.ingest.prices import PriceIngestor
        df = PriceIngestor(self.cache_root).fetch_prices(tickers, as_of, refresh)
        return df.reindex(columns=OUT_COLUMNS)


@dataclass
class ProbeResult:
    source: str
    ok: bool
    detail: str
    sample: pd.DataFrame | None = None


def probe(source: PriceSource, tickers: list[str], as_of: date) -> ProbeResult:
    """소량으로 소스를 검증한다. **대량 수집 전에 반드시 통과해야 한다.**

    알려지지 않은 API 를 추측으로 배선하고 2,598종목을 돌리면, 실패를 알아채는
    시점이 쿼터를 다 쓴 뒤가 된다(DART 키가 37자로 잘려 있던 것을 프로브로 알았다).
    """
    # 프로브가 곧 검증 주체다 — verified 게이트를 스스로 통과시킨다.
    # (게이트의 목적은 '검증 안 된 소스로 대량 수집'을 막는 것이지 프로브를 막는 게 아니다)
    had = getattr(source, "verified", None)
    if had is False:
        source.verified = True
    try:
        df = source.fetch(tickers, as_of)
    except Exception as e:                                        # noqa: BLE001
        if had is False:
            source.verified = False
        return ProbeResult(source.name, False, f"{type(e).__name__}: {e}"[:200])

    missing = [c for c in OUT_COLUMNS if c not in df.columns]
    if missing:
        return ProbeResult(source.name, False, f"컬럼 누락 {missing}")
    got = df.dropna(subset=["close"])
    if got.empty:
        return ProbeResult(source.name, False, "종가를 하나도 받지 못했다")
    if (got["close"] <= 0).any():
        return ProbeResult(source.name, False, "0 이하 종가가 있다")
    mark = getattr(source, "mark_verified", None)
    if callable(mark):
        mark(as_of, len(got))     # 다음 프로세스에서도 검증 상태가 유지되게
    return ProbeResult(source.name, True,
                       f"{len(got)}/{len(tickers)}종목 수신", got)


@dataclass
class Reconciliation:
    frame: pd.DataFrame
    primary: str
    compared_with: list[str] = field(default_factory=list)
    n_compared: int = 0
    n_disagree: int = 0

    def manifest(self) -> dict:
        return {"primary": self.primary, "compared_with": self.compared_with,
                "compared": self.n_compared, "disagree": self.n_disagree}


def reconcile(frames: dict[str, pd.DataFrame], primary: str,
              tolerance: float = PRICE_AGREEMENT_TOLERANCE) -> Reconciliation:
    """여러 소스의 종가를 대조한다.

    불일치해도 **값을 지우지 않는다.** primary 값을 그대로 두고
    `price_source_disagree` 플래그를 단다 — 어느 쪽이 맞는지 모르는 상태에서
    임의로 하나를 고르면 틀린 값을 확신을 갖고 쓰게 된다.
    판정은 하드가드가 한다(모르는 것은 통과시키지 않는다).

    소스가 하나뿐이면 플래그는 전부 False 다. **'검증했는데 이상 없음'이 아니라
    '검증하지 못했음'이므로** compared=0 을 매니페스트에 남긴다.
    """
    if primary not in frames:
        raise PriceSourceError(f"primary 소스 '{primary}' 의 결과가 없다")
    base = frames[primary].copy().reindex(columns=OUT_COLUMNS)
    base["price_source"] = primary
    base["price_source_disagree"] = False
    base["price_alt_close"] = pd.NA

    others = [k for k in frames if k != primary]
    if not others:
        return Reconciliation(base, primary)

    p = base.set_index("ticker")["close"].astype(float)
    disagree = pd.Series(False, index=p.index)
    alt = pd.Series(pd.NA, index=p.index, dtype="object")
    compared = pd.Series(False, index=p.index)

    for name in others:
        q = (frames[name].set_index("ticker")["close"]
             .astype(float).reindex(p.index))
        both = p.notna() & q.notna()
        compared |= both
        rel = (p - q).abs() / p.where(p != 0)
        hit = both & rel.gt(tolerance).fillna(False)
        disagree |= hit
        alt = alt.where(~hit, q)

    base["price_source_disagree"] = disagree.reindex(base["ticker"]).values
    base["price_alt_close"] = alt.reindex(base["ticker"]).values
    return Reconciliation(base, primary, others,
                          int(compared.sum()), int(disagree.sum()))
