"""KIND 업종명 → 내부 섹터 코드.

매핑 실패는 조용히 넘기지 않는다. sector_code=NULL 로 남기고 커버리지를 보고한다.
섹터가 비면 V2 섹터 정합성 검증이 그 종목에 대해 아무 일도 하지 않으므로,
'검증이 돌고 있다'는 착각을 만들지 않는 것이 중요하다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
MAP_PATH = REPO / "configs" / "sectors" / "kind_industry_map.yaml"
SECTOR_PATH = REPO / "configs" / "sectors" / "sector_map.yaml"


@dataclass(frozen=True)
class Rule:
    pattern: re.Pattern
    code: str


@lru_cache(maxsize=1)
def _load(map_path: str = "", sector_path: str = ""):
    cfg = yaml.safe_load(Path(map_path or MAP_PATH).read_text(encoding="utf-8"))
    sec = yaml.safe_load(Path(sector_path or SECTOR_PATH).read_text(encoding="utf-8"))
    universe = set(sec["codes"])

    rules = [Rule(re.compile(r["pattern"]), r["code"]) for r in cfg["rules"]]
    unknown = {r.code for r in rules} - universe
    if unknown:
        raise ValueError(f"kind_industry_map 에 sector_map 미등록 코드: {sorted(unknown)}")

    holding = [re.compile(p) for p in cfg.get("holding_name_patterns", [])]
    return rules, holding, universe


def map_industry(industry: str) -> str | None:
    """첫 매칭 규칙의 코드. 매칭 없으면 None."""
    if not industry:
        return None
    rules, _, _ = _load()
    for r in rules:
        if r.pattern.search(industry):
            return r.code
    return None


def looks_like_holding(name: str) -> bool:
    """지주사는 KSIC 상 '기타 금융업'/'경영 컨설팅'에 섞여 업종만으로 판별되지 않는다.
    상호 기반 휴리스틱이며, 정확한 판정은 아니다(플래그에 heuristic 임을 남길 것)."""
    _, holding, _ = _load()
    n = (name or "").strip()
    return any(p.search(n) for p in holding)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """corp_list 결과에 sector_code / is_holding_heuristic / unmapped_industry 부여."""
    out = df.copy()
    out["sector_code"] = out["industry"].map(map_industry)
    out["unmapped_industry"] = out["sector_code"].isna()
    out["is_holding"] = out["name"].map(looks_like_holding)
    out["is_financial"] = out["sector_code"].isin(
        ["BANK", "SECURITIES", "INSURANCE", "OTHER_FIN"])
    # 스팩은 상호로 판별 가능(관례상 '스팩'을 반드시 포함)
    out["is_spac"] = out["name"].str.contains("스팩", na=False)
    out["is_reit"] = out["name"].str.contains("리츠", na=False)
    return out


def coverage_report(df: pd.DataFrame) -> dict:
    """매핑 커버리지. 미매핑 업종을 종목수 순으로 보여준다."""
    n = len(df)
    unmapped = df[df["sector_code"].isna()]
    by_industry = (unmapped["industry"].value_counts().to_dict() if not unmapped.empty else {})
    return {
        "total": n,
        "mapped": int(n - len(unmapped)),
        "coverage": round((n - len(unmapped)) / n, 4) if n else 0.0,
        "unmapped_tickers": int(len(unmapped)),
        "unmapped_industries": by_industry,
        "sector_distribution": df["sector_code"].value_counts(dropna=False).to_dict(),
    }
