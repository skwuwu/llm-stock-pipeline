"""Evidence pack — LLM 이 볼 수 있는 세계의 전부.

원칙:
  1. **여기서 요약하지 않는다.** 요약본을 주면 V1 인용 검증(원문 대조)이 성립하지 않는다.
     길이가 문제면 섹션 발췌로 자르고, 무엇을 얼마나 버렸는지 meta 에 남긴다.
  2. 팩의 해시(pack_hash)가 LLM 캐시 키의 일부다. 같은 입력이면 재호출 비용 0.
  3. 소프트 플래그(일회성이익 의심 등)를 함께 넣어 LLM 이 맥락을 알게 한다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.enrich.segments import SegmentSet, parse_segments

SEGMENT_HINT_ROWS = 40   # 세그먼트 표로 보이는 줄만 추려 넣는다

# 사업보고서 본문에서 세그먼트/매출구성으로 보이는 줄
SEGMENT_KEYWORDS = ("부문", "세그먼트", "매출유형", "품목", "매출 비중", "매출액 비중")


@dataclass
class EvidencePack:
    ticker: str
    name: str
    sector_code: str | None
    as_of: str
    business: str
    business_meta: dict
    segments_text: str          # V1 인용 대조용 원문 줄
    segments: dict              # V3 대조용 구조화 비중
    disclosures: list[dict]
    metrics: dict
    flags: list[str]
    pack_hash: str = ""

    def finalize(self) -> "EvidencePack":
        h = hashlib.sha256()
        for part in (self.ticker, self.business, self.segments_text,
                     json.dumps(self.disclosures, ensure_ascii=False, sort_keys=True)):
            h.update(part.encode())
            h.update(b"\x00")
        self.pack_hash = h.hexdigest()[:32]
        return self

    def to_cascade_input(self) -> dict:
        """cascade.narrow_candidates / build_user 가 기대하는 형태."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "sector_code": self.sector_code,
            "business": self.business,
            "segments_text": self.segments_text,
            "segments": self.segments,
            "disclosures": self.disclosures,
            "pack_hash": self.pack_hash,
            "theme_segment_share": {},   # V3 는 수치 세그먼트가 붙은 뒤에 활성화
        }

    def write(self, root: Path) -> Path:
        d = root / self.ticker
        d.mkdir(parents=True, exist_ok=True)
        (d / "business.md").write_text(self.business, encoding="utf-8")
        (d / "pack.json").write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return d


# 지표 중 LLM 에 넘기는 것. 주가·밸류에이션은 넘기지 않는다 —
# LLM 이 투자 판단을 하게 만들면 사업 분류라는 임무에서 벗어난다.
METRIC_KEYS = ["revenue_ttm", "operating_income_ttm", "net_income_ttm", "sector_code"]

FLAG_KEYS = ["oneoff_profit_suspect", "per_op_divergence", "minority_interest_large",
             "cfs_missing_used_ofs", "capex_unmapped", "non_standard_fiscal_month",
             "ni_fallback_used", "equity_fallback_used", "holding_company",
             "financial_sector"]


def extract_segment_lines(business: str, limit: int = SEGMENT_HINT_ROWS) -> str:
    """본문에서 세그먼트·매출구성으로 보이는 줄만 추린다.

    수치 파싱은 하지 않는다. 잘못 파싱한 숫자를 LLM 에 넘기느니
    원문 줄을 그대로 넘기는 편이 V1 인용 검증과도 맞는다.
    """
    hits = []
    for line in business.splitlines():
        t = line.strip()
        if not t or len(t) > 300:
            continue
        if any(k in t for k in SEGMENT_KEYWORDS) and ("%" in t or "|" in t):
            hits.append(t)
            if len(hits) >= limit:
                break
    return "\n".join(hits)


def build_pack(row: pd.Series, business: str, business_meta: dict,
               disclosures: list[dict], as_of: date,
               segment_source: str | None = None) -> EvidencePack:
    """business 는 LLM 입력용(예산 제한), segment_source 는 세그먼트 추출용(전체 본문).

    LLM 입력 예산이 세그먼트 추출을 제약할 이유가 없다 — 추출은 로컬 연산이다.
    """
    metrics = {k: (None if pd.isna(row.get(k)) else _plain(row.get(k)))
               for k in METRIC_KEYS if k in row}
    flags = [k for k in FLAG_KEYS if bool(row.get(k))]
    rev = row.get("revenue_ttm")
    rev = None if rev is None or pd.isna(rev) else float(rev)
    seg = parse_segments(segment_source or business, revenue=rev)
    return EvidencePack(
        ticker=str(row["ticker"]),
        name=str(row.get("name") or ""),
        sector_code=(None if pd.isna(row.get("sector_code")) else row.get("sector_code")),
        as_of=as_of.isoformat(),
        business=business,
        business_meta=business_meta,
        segments_text=(seg.source_line or extract_segment_lines(business)),
        segments=seg.as_dict(),
        disclosures=disclosures,
        metrics=metrics,
        flags=flags,
    ).finalize()


def _plain(v):
    if isinstance(v, (int, float)):
        return float(v)
    return str(v)
