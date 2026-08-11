"""시세 + 주식수 인제스천.

경로 선택 근거:
  - 시세: FinanceDataReader. KRX 데이터포털이 로그인을 요구하게 되어 pykrx 가 막혔고,
    FDR 의 DataReader 는 다른 백엔드를 써서 살아 있다.
  - 주식수: DART stockTotqySttus. 보통주/우선주/자기주식이 분리돼 있어
    시총을 우리가 직접, 정확한 기준으로 계산할 수 있다.
    벤더 시총을 그대로 쓰는 것보다 낫다 — 어떤 주식수를 썼는지 알 수 있으니까.

시총 = 종가 × 발행주식수. PER 분자는 보통주 + 우선주 합.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pipeline.ingest.dart import DartClient
from pipeline.ingest.krx import PRICE_COLUMNS

ADTV_WINDOW = 20

# ── 주식 종류 라벨 분류 ──────────────────────────────────────────────
# DART stockTotqySttus 의 `se` 는 **열거형이 아니라 자유 텍스트다.** 실측
# 2,704개 원천에서 130가지 넘는 표기가 나왔다:
#     보통주 / 보통주식 / 기명식보통주 / 의결권 있는 주식 / 의결권 있는 주식(보통주)
#     우선주 / 종류주식 / 1종 종류주식 / 무의결권부우선주 / 배당우선전환주식 …
#     보톧주 / 보퉁주 / 보통부              ← 오타까지 있다
#
# 완전일치로 '보통주'만 보다가 나머지를 통째로 놓쳤다. 주식수가 NULL 이 되면
# 시총이 만들어지지 않고, 하드가드의 below_min_market_cap_or_missing 으로
# **조용히 배제**된다 — 실측 230종목(8.9%)이 그 상태였고 삼성전기·LG전자·
# 한미반도체·셀트리온·HD현대일렉트릭 같은 대형주가 들어 있었다.
# '조용히'가 문제의 핵심이다. 배제 사유가 '시총 미달'로 찍혀 데이터 결손처럼
# 보이지 않았다.
_SKIP_LABELS = frozenset({"합계", "비고", "기타", "기타주", "기타주식",
                          "유통주식", "주식수", "자본주", "종류의 주식"})
# 접두·부분 매칭은 순서가 곧 규칙이다. 우선주 계열을 **먼저** 걸러야
# '의결권 있는 주식(우선주)' 같은 혼합 표기가 보통주로 새지 않는다.
_OTHER_MARKS = ("우선", "종류주", "전환주", "상환주", "신형")
_COMMON_MARKS = ("보통", "보톧", "보퉁")     # 뒤 둘은 실측된 오타


def classify_share_class(se: str) -> str | None:
    """`se` 라벨 → 'common' | 'other' | None(집계행·판정불가).

    None 을 돌려주는 것과 'other' 는 다르다 — 전자는 그 행을 무시하라는 뜻이고
    후자는 우선주/종류주로 세라는 뜻이다.
    """
    s = " ".join((se or "").split())          # 줄바꿈·중복공백 정규화
    if not s or s in _SKIP_LABELS or "합계" in s:
        return None
    if any(k in s for k in _OTHER_MARKS):
        return "other"
    if any(k in s for k in _COMMON_MARKS):
        return "common"
    if "의결권" in s:
        # '의결권 없는 주식' → 종류주, '의결권 있는 주식' → 보통주
        return "other" if ("없" in s or "무의결" in s) else "common"
    return None


class PriceFetchError(RuntimeError):
    pass


@dataclass
class PriceIngestor:
    cache_root: Path
    workers: int = 8
    lookback_days: int = 45      # 20 거래일 확보용 여유

    def __post_init__(self) -> None:
        self.cache_root = Path(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # ── 시세 ─────────────────────────────────────────────────────
    def _one(self, ticker: str, start: date, end: date) -> dict | None:
        import FinanceDataReader as fdr
        try:
            df = fdr.DataReader(ticker, start, end)
        except Exception:                                        # noqa: BLE001
            return None
        if df is None or df.empty or "Close" not in df:
            return None
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        last = df.iloc[-1]
        # 거래대금 근사: 거래량 × 종가. FDR 은 거래대금을 직접 주지 않는다.
        # 실제 거래대금은 체결가별 합이라 소폭 차이가 있다(유동성 필터 용도로는 충분).
        adtv = float((df["Volume"] * df["Close"]).tail(ADTV_WINDOW).mean())
        return {"ticker": ticker, "close": float(last["Close"]),
                "adtv_20d": adtv, "price_date": df.index[-1].date()}

    def fetch_prices(self, tickers: list[str], as_of: date,
                     refresh: bool = False) -> pd.DataFrame:
        """날짜별 증분 캐시.

        캐시 키를 날짜만으로 두면 **다른 종목 집합으로 만든 캐시가 조용히 재사용**되어
        요청한 것보다 적은 결과가 완전한 것처럼 돌아온다. 그래서 캐시에 없는 티커만
        가져와 병합하고, 요청 집합으로 잘라서 반환한다.
        """
        cache = self.cache_root / f"close_{as_of.isoformat()}.parquet"
        cached = pd.read_parquet(cache) if (cache.exists() and not refresh) else \
            pd.DataFrame(columns=["ticker", "close", "adtv_20d", "price_date"])

        want = list(dict.fromkeys(tickers))
        have = set(cached["ticker"]) if not cached.empty else set()
        todo = [t for t in want if t not in have]

        rows, failed = [], []
        if todo:
            start = as_of - timedelta(days=self.lookback_days)
            with ThreadPoolExecutor(max_workers=self.workers) as ex:
                futs = {ex.submit(self._one, t, start, as_of): t for t in todo}
                for f in as_completed(futs):
                    r = f.result()
                    (rows.append(r) if r else failed.append(futs[f]))

        merged = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True) \
            if rows else cached
        if merged.empty:
            raise PriceFetchError(
                f"{as_of}: 시세를 한 건도 못 받았다({len(want)}종목 시도). "
                f"소스 차단 여부를 확인할 것 — 빈 결과로 통과시키지 않는다.")
        merged = merged.drop_duplicates(subset=["ticker"], keep="last")
        cache.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache, index=False)

        out = merged[merged["ticker"].isin(want)].reset_index(drop=True)
        out.attrs["failed"] = failed
        out.attrs["from_cache"] = len(want) - len(todo)
        return out

    # ── 주식수 ───────────────────────────────────────────────────
    @staticmethod
    def _num(s) -> int | None:
        t = str(s or "").strip().replace(",", "")
        if t in {"", "-"}:
            return None
        try:
            return int(float(t))
        except ValueError:
            return None

    def fetch_shares(self, client: DartClient, corp_map: dict[str, str],
                     year: int, quarter: int) -> pd.DataFrame:
        """corp_map: ticker -> corp_code. 반환: ticker, shares_common,
        shares_preferred, treasury_shares, shares_asof"""
        rows = []
        for ticker, corp_code in corp_map.items():
            if not corp_code:
                continue
            try:
                p = client.shares_outstanding(corp_code, year, quarter)
            except Exception:                                    # noqa: BLE001
                continue
            if p.get("status") != "000":
                continue
            common = pref = treasury = None
            asof = None
            for r in p.get("list", []):
                kind = classify_share_class(r.get("se"))
                if kind == "common":
                    common = self._num(r.get("istc_totqy"))
                    treasury = self._num(r.get("tesstk_co"))
                    asof = (r.get("stlm_dt") or "").strip() or None
                elif kind == "other":
                    # 종류주가 여러 줄로 쪼개져 오는 경우가 있다(1종/2종/3종).
                    # 덮어쓰면 마지막 줄만 남아 우선주 시총이 과소계상된다.
                    v = self._num(r.get("istc_totqy"))
                    if v is not None:
                        pref = (pref or 0) + v
            if common is None:
                continue
            rows.append({"ticker": ticker, "shares_common": common,
                         "shares_preferred": pref or 0,
                         "treasury_shares": treasury or 0, "shares_asof": asof})
        return pd.DataFrame(rows)


def build_price_table(prices: pd.DataFrame, shares: pd.DataFrame,
                      as_of: date) -> pd.DataFrame:
    """종가 × 주식수 → 시총. 주식수가 없으면 시총을 만들지 않는다(0으로 메우지 않음).

    시총 결측은 하드 가드(below_min_market_cap_or_missing)에서 배제된다.
    """
    df = prices.merge(shares, on="ticker", how="left")
    df["date"] = as_of
    df["market_cap_common"] = df["close"] * df["shares_common"]
    df["market_cap_total"] = df["close"] * (
        df["shares_common"].fillna(0) + df["shares_preferred"].fillna(0))
    df.loc[df["shares_common"].isna(), ["market_cap_common", "market_cap_total"]] = None
    for c in ("shares_common", "shares_preferred", "treasury_shares"):
        df[c] = df[c].astype("Int64")
    return df.reindex(columns=PRICE_COLUMNS)
