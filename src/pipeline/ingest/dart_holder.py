"""최대주주 및 특수관계인 현황 (DART hyslrSttus) → PIT facts.

지배구조 지표로 의미 있는 것은 개인 최대주주 한 명이 아니라
**최대주주 및 특수관계인 합계**다. 응답에는 사람/법인별 행 뒤에 `계` 행이 붙는다.

배당과 같은 PIT 규율: reported_at 은 결산일이 아니라 사업보고서 접수일이다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

BASE = "https://opendart.fss.or.kr/api/hyslrSttus.json"


class HolderFetchError(RuntimeError):
    pass


def _num(s: str | None) -> float | None:
    t = str(s or "").replace(",", "").strip()
    if t in ("-", "", "N/A"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


@dataclass
class HolderClient:
    raw_root: Path
    api_key: str | None = None
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        import os
        from pipeline.ingest.dart import _dotenv_key, validate_dart_key
        self.api_key = self.api_key or os.environ.get("DART_API_KEY") or _dotenv_key()
        if self.api_key:
            validate_dart_key(self.api_key, "HolderClient")
        self.raw_root = Path(self.raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.calls = 0

    def fetch(self, corp_code: str, year: int, reprt: str = "11011",
              refresh: bool = False) -> dict:
        p = self.raw_root / f"{corp_code}_{year}_{reprt}.json"
        if p.exists() and not refresh:
            return json.loads(p.read_text(encoding="utf-8"))
        if not self.api_key:
            raise HolderFetchError(f"DART_API_KEY 미설정이고 캐시도 없다: {p}")
        r = self.session.get(BASE, params={"crtfc_key": self.api_key,
                                           "corp_code": corp_code,
                                           "bsns_year": str(year),
                                           "reprt_code": reprt},
                             timeout=self.timeout_s)
        r.raise_for_status()
        self.calls += 1
        d = r.json()
        if d.get("status") not in ("000", "013"):
            raise HolderFetchError(
                f"{corp_code} {year}: DART status={d.get('status')} {d.get('message')}")
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        return d


def parse_holders(raw: dict, ticker: str, lag_days: int = 1) -> list[dict]:
    """raw → facts 행. 데이터가 없으면 빈 목록('지분 0'이 아니라 '모름')."""
    rows = raw.get("list") or []
    if raw.get("status") == "013" or not rows:
        return []

    rcept = str(rows[0].get("rcept_no") or "")
    if len(rcept) < 8 or not rcept[:8].isdigit():
        raise HolderFetchError(f"{ticker}: rcept_no 가 없어 PIT 시점을 정할 수 없다")
    reported = datetime.strptime(rcept[:8], "%Y%m%d").date()

    common = [r for r in rows if (r.get("stock_knd") or "").strip() in ("보통주", "-", "")]
    total_rows = [r for r in common if re.sub(r"\s", "", r.get("nm") or "") in ("계", "합계")]

    if total_rows:
        pct = _num(total_rows[0].get("trmend_posesn_stock_qota_rt"))
    else:
        # `계` 행이 없으면 개별 행을 더한다. 다만 합산은 추정이므로
        # 행이 하나도 없을 때 0 을 만들지는 않는다.
        vals = [_num(r.get("trmend_posesn_stock_qota_rt")) for r in common]
        vals = [v for v in vals if v is not None]
        pct = sum(vals) if vals else None
    if pct is None:
        return []

    largest = max((_num(r.get("trmend_posesn_stock_qota_rt")) or 0.0)
                  for r in common
                  if re.sub(r"\s", "", r.get("nm") or "") not in ("계", "합계")) \
        if len(common) > len(total_rows) else pct

    fiscal_end = _parse_stlm(rows[0].get("stlm_dt"), reported)
    base = {"ticker": ticker, "fiscal_end_date": fiscal_end,
            "reported_at": reported,
            "available_at": reported + timedelta(days=lag_days),
            "source_doc_id": rcept}
    return [
        {**base, "element": "OWNER_STAKE_PCT", "value": float(pct)},
        {**base, "element": "LARGEST_HOLDER_PCT", "value": float(largest)},
    ]


def _parse_stlm(s: str | None, reported: date) -> date:
    m = re.match(r"(\d{4})[-.년\s]+(\d{1,2})[-.월\s]*(\d{1,2})?", (s or "").strip())
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if d == 0:
            d = 31 if mo in (1, 3, 5, 7, 8, 10, 12) else 30
        try:
            return date(y, mo, d)
        except ValueError:
            pass
    return date(reported.year - 1, 12, 31)
