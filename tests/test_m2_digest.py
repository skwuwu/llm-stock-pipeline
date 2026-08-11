"""다이제스트 렌더러. 데이터가 없거나 검증이 전멸해도 조용히 그럴듯한 문서를 내면 안 된다."""

from __future__ import annotations

from datetime import date

import pandas as pd
import yaml
from pathlib import Path

from pipeline.report.digest import render

REPO = Path(__file__).resolve().parents[1]
TAXONOMY = yaml.safe_load((REPO / "configs/themes/taxonomy_v1.yaml").read_text("utf-8"))

SURV = pd.DataFrame([
    {"ticker": "000001", "name": "가전기", "sector_code": "ELEC_EQUIP",
     "per": 6.2, "pbr": 0.71, "fcf_yield": 0.091},
    {"ticker": "000002", "name": "나화학", "sector_code": "CHEM",
     "per": 8.0, "pbr": 0.60, "fcf_yield": 0.050},
])
FUNNEL = {"eligible": 500, "final": 2, "universe": 2700}
QUALITY = {"soft": {"oneoff_profit_suspect": 3, "capex_unmapped": 0}}


def _verdicts(rows):
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def test_verified_core_appears_in_body_with_citation():
    v = _verdicts([{"ticker": "000001", "theme_id": "ai_datacenter_power", "role": "core",
                    "status": "verified", "flags": [], "reject_reason": None,
                    "confidence": 0.9, "share_evidence": "confirmed",
                    "actual_share": 0.72, "claimed_share": 0.72,
                    "rationale": "초고압 변압기 주력", "evidence_quote": "초고압 변압기를 제조"}])
    md = render(date(2026, 8, 6), SURV, v, TAXONOMY, FUNNEL, QUALITY,
                {"status": {"verified": 1}, "verified_core": 1, "assignments": 1,
                 "hallucinated_citation_rate": 0.0})
    assert "AI 데이터센터 전력기기" in md
    assert "초고압 변압기를 제조" in md          # 근거가 본문에 함께 실린다
    # 투자 판단 언어는 본문에 없어야 한다(면책 문구는 예외)
    body = md.split("---")[0]
    assert "매수" not in body and "목표주가" not in body


def test_rejected_assignment_stays_out_of_body():
    """폐기된 배정이 본문에 실리면 검증 레이어가 무의미하다."""
    v = _verdicts([{"ticker": "000002", "theme_id": "ev_battery_materials", "role": "core",
                    "status": "rejected", "flags": [],
                    "reject_reason": "hallucinated_citation",
                    "confidence": 0.95, "share_evidence": "unavailable",
                    "actual_share": None, "claimed_share": None,
                    "rationale": "양극재 신사업",
                    "evidence_quote": "존재하지 않는 문장"}])
    md = render(date(2026, 8, 6), SURV, v, TAXONOMY, FUNNEL, QUALITY,
                {"status": {"rejected": 1}, "verified_core": 0, "assignments": 1,
                 "hallucinated_citation_rate": 1.0})
    body = md.split("## 부록 A")[0]
    assert "양극재 신사업" not in body            # 본문 아님
    assert "hallucinated_citation" in md          # 부록에는 남는다


def test_empty_verdicts_says_so_rather_than_faking_content():
    md = render(date(2026, 8, 6), SURV, pd.DataFrame(), TAXONOMY, FUNNEL, QUALITY,
                {"status": {}, "assignments": 0})
    assert "검증을 통과한 테마 배정이 없다" in md


def test_hallucination_rate_is_surfaced_in_summary():
    """인용 실패율은 이 파이프라인의 핵심 지표라 요약에 항상 보여야 한다."""
    md = render(date(2026, 8, 6), SURV, pd.DataFrame(), TAXONOMY, FUNNEL, QUALITY,
                {"tiers": {}, "assignments": 0, "hallucinated_citation_rate": 0.123})
    assert "12.3%" in md


def test_disclaimer_present():
    md = render(date(2026, 8, 6), SURV, pd.DataFrame(), TAXONOMY, FUNNEL, QUALITY, {})
    assert "매수 신호가 아니다" in md
