"""배당에 관한 사항 (DART alotMatter) → PIT facts.

별도 테이블을 만들지 않고 facts_financial 에 element 로 넣는다. 그래야
available_at / revision_of / renormalize 같은 PIT 기계장치를 그대로 쓴다.

**PIT 규율이 특히 중요한 데이터다.** 배당은 정기주총에서 확정되고 사업보고서로
공시된다. 결산일(2025-12-31) 기준으로 배당을 안다고 두면, 실제로는 이듬해 3월에야
알 수 있었던 것을 미리 아는 룩어헤드가 된다. 그래서 reported_at 은 결산일이 아니라
**사업보고서 접수일**(rcept_no 앞 8자리)이다.

수집하지 못한 것과 배당이 없는 것을 구분한다:
  응답 있음 + 주당배당금 '-'  → 무배당 (DPS = 0). 사실이다.
  응답 자체가 없음            → 미수집. 값을 만들지 않는다(NaN).
이 구분이 없으면 미수집 종목이 '무배당'으로 둔갑해 스크린에서 조용히 탈락한다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://opendart.fss.or.kr/api/alotMatter.json"

# se(구분) 텍스트 → canonical element. 공백·괄호 표기가 흔들려 정규화 후 비교한다.
_SE_MAP = {
    "주당현금배당금(원)": ("DPS_CASH", "보통주"),
    "(연결)현금배당성향(%)": ("PAYOUT_RATIO_PCT", None),
    "현금배당수익률(%)": ("DIV_YIELD_REPORTED_PCT", "보통주"),
    "현금배당금총액(백만원)": ("DIVIDEND_TOTAL_MM", None),
}
NO_VALUE = {"-", "", "0", None}


class DividendFetchError(RuntimeError):
    pass


def _norm_se(s: str) -> str:
    return re.sub(r"\s", "", s or "")


def _num(s: str | None) -> float | None:
    if s is None:
        return None
    t = str(s).replace(",", "").strip()
    if t in ("-", "", "N/A"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


@dataclass
class DividendClient:
    raw_root: Path
    api_key: str | None = None
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        import os
        from pipeline.ingest.dart import _dotenv_key, validate_dart_key
        self.api_key = self.api_key or os.environ.get("DART_API_KEY") or _dotenv_key()
        if self.api_key:
            validate_dart_key(self.api_key, "DividendClient")
        self.raw_root = Path(self.raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.calls = 0

    def fetch(self, corp_code: str, year: int, reprt: str = "11011",
              refresh: bool = False) -> dict:
        """raw 응답. 접수번호별이 아니라 (corp, year, reprt) 별로 불변 보관한다."""
        p = self.raw_root / f"{corp_code}_{year}_{reprt}.json"
        if p.exists() and not refresh:
            return json.loads(p.read_text(encoding="utf-8"))
        if not self.api_key:
            raise DividendFetchError(f"DART_API_KEY 미설정이고 캐시도 없다: {p}")
        r = self.session.get(BASE, params={"crtfc_key": self.api_key,
                                           "corp_code": corp_code,
                                           "bsns_year": str(year),
                                           "reprt_code": reprt},
                             timeout=self.timeout_s)
        r.raise_for_status()
        self.calls += 1
        d = r.json()
        status = d.get("status")
        # 013 = 조회된 데이터가 없음. 정상적인 결과이므로 캐시해 재호출을 막는다.
        if status not in ("000", "013"):
            raise DividendFetchError(
                f"{corp_code} {year}: DART status={status} {d.get('message')}")
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return d


def parse_dividend(raw: dict, ticker: str, lag_days: int = 1) -> list[dict]:
    """raw → facts 행. 데이터가 없으면 빈 목록(무배당이 아니라 '모름')."""
    rows = raw.get("list") or []
    if raw.get("status") == "013" or not rows:
        return []

    rcept = str(rows[0].get("rcept_no") or "")
    if len(rcept) < 8 or not rcept[:8].isdigit():
        raise DividendFetchError(f"{ticker}: rcept_no 가 없어 PIT 시점을 정할 수 없다")
    reported = datetime.strptime(rcept[:8], "%Y%m%d").date()

    # 결산일. stlm_dt 가 'YYYY-MM-DD' 또는 'YYYY년 MM월' 형태로 온다.
    fiscal_end = _parse_stlm(rows[0].get("stlm_dt"), reported)

    picked: dict[str, float] = {}
    for r in rows:
        key = _SE_MAP.get(_norm_se(r.get("se")))
        if not key:
            continue
        element, want_knd = key
        if want_knd and (r.get("stock_knd") or "").strip() not in (want_knd, "-", ""):
            continue          # 우선주 행은 건너뛴다 — 스크린 대상은 보통주다
        v = _num(r.get("thstrm"))
        if element not in picked:
            # 보고는 됐는데 값이 '-' 인 것은 **무배당**이다. 0 으로 확정한다.
            picked[element] = 0.0 if v is None else v

    out = []
    for element, value in picked.items():
        out.append({
            "ticker": ticker, "element": element, "value": float(value),
            "fiscal_end_date": fiscal_end,
            "reported_at": reported,
            "available_at": reported + timedelta(days=lag_days),
            "source_doc_id": rcept,
        })
    return out


def _parse_stlm(s: str | None, reported: date) -> date:
    """결산일 파싱. 실패하면 접수 연도의 직전 결산으로 두되 추정임을 감춘다 —
    이 값은 PIT 판정에 쓰이지 않는다(판정은 available_at 이 한다)."""
    t = (s or "").strip()
    m = re.match(r"(\d{4})[-.년\s]+(\d{1,2})[-.월\s]*(\d{1,2})?", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if d == 0:
            d = 31 if mo in (1, 3, 5, 7, 8, 10, 12) else 30
        try:
            return date(y, mo, d)
        except ValueError:
            pass
    return date(reported.year - 1, 12, 31)
