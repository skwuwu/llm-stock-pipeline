"""M2 검증 레이어. LLM 호출 없이 돈다."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline.ingest.dart_document import extract_business_section, to_plain_text
from pipeline.verify.layers import (
    SectorMatrix, verification_metrics, verify_assignment, verify_citation,
)

PACK = {
    "ticker": "000001",
    "business": ("당사는 초고압 변압기와 가스절연개폐장치(GIS)를 제조하며, "
                 "북미 전력청 및 데이터센터향 수주가 증가하고 있습니다.\n"
                 "배전반 사업도 영위합니다."),
    "segments_text": "중전기기 부문 매출 비중 72% | 배전 부문 28%",
    "segments": {"complete": True, "corroborated": True,
                 "source_line": "중전기기 부문 매출 비중 72% | 배전 부문 28%",
                 "segments": [{"name": "중전기기", "share": 0.72, "amount": None},
                              {"name": "배전", "share": 0.28, "amount": None}]},
    "disclosures": [{"date": "2026-07-01", "title": "단일판매·공급계약 체결(변압기)"}],
}

# 세그먼트 수치를 못 뽑은 회사 — V3 를 '확인 불가'로 처리해야 한다
PACK_NO_SEG = {**PACK, "segments": {}, "segments_text": ""}


def _asg(**kw):
    base = {"theme_id": "ai_datacenter_power", "role": "core",
            "rationale": "초고압 변압기 주력", "evidence_source": "business",
            "evidence_quote": "초고압 변압기와 가스절연개폐장치(GIS)를 제조",
            "revenue_share_claim": 0.72, "segment_name": "중전기기",
            "confidence": 0.9}
    return {**base, **kw}


# ── V1 인용 검증 ────────────────────────────────────────────────────
def test_exact_citation_passes():
    assert verify_citation("초고압 변압기와 가스절연개폐장치(GIS)를 제조", PACK, "business")


def test_whitespace_differences_tolerated():
    assert verify_citation("초고압  변압기와\n가스절연개폐장치(GIS)를 제조", PACK, "business")


def test_hallucinated_citation_rejected():
    assert not verify_citation("당사는 2차전지 양극재를 생산합니다", PACK, "business")


def test_too_short_citation_rejected():
    """짧은 인용은 우연히 맞을 수 있어 근거로 인정하지 않는다."""
    assert not verify_citation("제조", PACK, "business")


def test_citation_in_segments_and_disclosures():
    assert verify_citation("중전기기 부문 매출 비중 72%", PACK, "segments")
    assert verify_citation("단일판매·공급계약 체결(변압기)", PACK, "disclosures")


# ── V2 섹터 정합성 ──────────────────────────────────────────────────
def test_sector_matrix_allows_and_conflicts():
    m = SectorMatrix()
    assert not m.conflicts("ai_datacenter_power", "ELEC_EQUIP")
    assert m.conflicts("ai_datacenter_power", "FOOD_BEV")


def test_unknown_sector_is_not_a_conflict():
    """섹터 미상이면 판정하지 않는다 — 모르는 것과 어긋난 것은 다르다."""
    assert not SectorMatrix().conflicts("ai_datacenter_power", None)


def test_any_sector_theme_never_conflicts():
    m = SectorMatrix()
    assert not m.conflicts("value_up_governance", "FOOD_BEV")


# ── 종합 판정 ───────────────────────────────────────────────────────
def test_clean_core_assignment_is_verified_and_confirmed():
    v = verify_assignment(_asg(), PACK, SectorMatrix(), "ELEC_EQUIP")
    assert v.status == "verified" and not v.flags and v.reject_reason is None
    assert v.share_evidence == "confirmed"


def test_hallucination_rejected_regardless_of_confidence():
    v = verify_assignment(_asg(evidence_quote="당사는 원전 주기기를 제작합니다",
                               confidence=0.99),
                          PACK, SectorMatrix(), "ELEC_EQUIP")
    assert v.status == "rejected" and v.reject_reason == "hallucinated_citation"


def test_unknown_theme_rejected():
    v = verify_assignment(_asg(theme_id="made_up_theme"), PACK, SectorMatrix(), "ELEC_EQUIP")
    assert v.status == "rejected" and v.reject_reason == "unknown_theme"


def test_sector_conflict_flags_not_rejects():
    """신규 사업 진출은 실재하므로 폐기하지 않는다. 표시만 남긴다."""
    v = verify_assignment(_asg(), PACK, SectorMatrix(), "FOOD_BEV")
    assert v.reject_reason is None
    assert "sector_conflict" in v.flags
    assert v.status == "verified"


def test_low_claim_downgrades_core_when_numbers_unavailable():
    """세그먼트 수치가 없으면 LLM 주장으로라도 강등한다(차선책).
    수치가 있을 때는 주장이 아니라 수치로 판정한다 —
    test_actual_share_below_floor_downgrades_core 참조."""
    v = verify_assignment(_asg(revenue_share_claim=0.02, segment_name=None),
                          PACK_NO_SEG, SectorMatrix(), "ELEC_EQUIP")
    assert v.role == "adjacent"
    assert any(f.startswith("role_downgraded") for f in v.flags)


def test_core_without_share_evidence_flagged():
    """비중 근거가 아무것도 없는 core 배정."""
    v = verify_assignment(_asg(revenue_share_claim=0.0, segment_name=None),
                          PACK_NO_SEG, SectorMatrix(), "ELEC_EQUIP")
    assert "core_without_share_evidence" in v.flags


def test_wrong_source_but_real_quote_is_salvaged():
    """출처만 틀렸고 원문에는 있는 경우 폐기하지 않고 플래그만 남긴다."""
    v = verify_assignment(_asg(evidence_source="disclosures"), PACK,
                          SectorMatrix(), "ELEC_EQUIP")
    assert v.reject_reason is None
    assert any(f.startswith("citation_source_mismatch") for f in v.flags)


def test_metrics_shape():
    m = SectorMatrix()
    vs = [verify_assignment(_asg(), PACK, m, "ELEC_EQUIP"),
          verify_assignment(_asg(evidence_quote="없는 문장입니다 정말로"), PACK, m, "ELEC_EQUIP")]
    k = verification_metrics(vs, n_stocks=1)
    assert k["assignments"] == 2
    assert k["hallucinated_citation_rate"] == 0.5
    assert k["status"] == {"verified": 1, "rejected": 1}
    assert k["verified_core"] == 1
    assert k["share_evidence"] == {"confirmed": 1}


# ── 사업보고서 섹션 추출 ────────────────────────────────────────────
# 실제 DART 마크업처럼 속성을 붙여둔다 — 속성 안의 값이 본문에 새면 안 된다
MARKUP = """<DOCUMENT><P>I. 회사의 개요</P><P>회사 일반 정보입니다.</P>
<P USERMARK="F-16 B">II. 사업의 내용</P><P ACLASS="NORMAL">당사는 초고압 변압기를 제조합니다.</P>
<TABLE W="500"><TR><TE ALIGN="C">부문</TE><TE>매출</TE></TR><TR><TE>중전기기</TE><TE>72%</TE></TR></TABLE>
<P>III. 재무에 관한 사항</P><P>재무제표 내용입니다.</P></DOCUMENT>"""


def test_extracts_business_section_only():
    ex = extract_business_section(MARKUP)
    assert ex.found_section
    assert "초고압 변압기를 제조" in ex.text
    assert "재무제표 내용" not in ex.text
    assert "회사 일반 정보" not in ex.text


def test_tag_attributes_never_leak_into_text():
    """<P USERMARK="F-16 B"> 처럼 속성이 붙은 태그에서 속성값이 본문에 남으면 안 된다.
    남으면 LLM 이 그걸 사업 설명으로 읽고, V1 인용 검증도 오염된다."""
    t = to_plain_text(MARKUP)
    for frag in ("USERMARK", "ACLASS", "ALIGN", "<", ">"):
        assert frag not in t, f"태그 잔여물: {frag}"


def test_table_cells_survive_as_text():
    """세그먼트 수치가 표 안에 있으므로 셀 구분자를 남겨야 한다."""
    assert "중전기기" in to_plain_text(MARKUP)
    assert "72%" in to_plain_text(MARKUP)


def test_missing_section_is_flagged_not_silent():
    ex = extract_business_section("<P>사업 관련 서술이 없는 문서</P>")
    assert not ex.found_section       # 조용히 엉뚱한 본문을 넘기지 않는다


def test_truncation_records_how_much_was_dropped():
    """요약하지 않고 자르되, 얼마나 버렸는지 남긴다."""
    long = MARKUP.replace("당사는 초고압 변압기를 제조합니다.", "가" * 30_000)
    ex = extract_business_section(long, max_chars=1_000)
    assert len(ex.text) == 1_000
    assert ex.truncated_chars > 0


# ── Evidence pack ───────────────────────────────────────────────────
def test_pack_roundtrip_and_hash_stability(tmp_path):
    """pack_hash 는 LLM 캐시 키의 일부다 — 같은 입력이면 같아야 재호출이 0원이 된다."""
    import pandas as pd
    from datetime import date as _date
    from pipeline.enrich.evidence import EvidencePack, build_pack

    row = pd.Series({"ticker": "000001", "name": "가전기", "sector_code": "ELEC_EQUIP",
                     "revenue_ttm": 1.0e12, "operating_income_ttm": 1.0e11,
                     "net_income_ttm": 8.0e10, "oneoff_profit_suspect": True,
                     "capex_unmapped": False})
    meta = {"rcept_no": "20250311001085", "found_section": True,
            "truncated_chars": 0, "total_chars": 500}
    body = "당사는 초고압 변압기를 제조합니다.\n중전기기 부문 매출 비중 72% | 배전 28%"

    p1 = build_pack(row, body, meta, [{"date": "2026-07-01", "title": "공급계약"}], _date(2026, 8, 6))
    p2 = build_pack(row, body, meta, [{"date": "2026-07-01", "title": "공급계약"}], _date(2026, 8, 6))
    assert p1.pack_hash == p2.pack_hash and len(p1.pack_hash) == 32

    p3 = build_pack(row, body + " 추가 문장.", meta, [], _date(2026, 8, 6))
    assert p3.pack_hash != p1.pack_hash        # 본문이 바뀌면 캐시가 무효화돼야 한다

    assert "oneoff_profit_suspect" in p1.flags
    assert "capex_unmapped" not in p1.flags
    assert "매출 비중 72%" in p1.segments_text

    d = p1.write(tmp_path)
    restored = EvidencePack(**__import__("json").loads((d / "pack.json").read_text("utf-8")))
    assert restored.to_cascade_input()["pack_hash"] == p1.pack_hash


def test_pack_excludes_valuation_from_llm_input():
    """주가·밸류에이션을 LLM 에 넘기면 사업 분류가 아니라 투자 판단을 하게 된다."""
    from pipeline.enrich.evidence import METRIC_KEYS
    for banned in ("per", "pbr", "close", "market_cap_used", "fcf_yield"):
        assert banned not in METRIC_KEYS


# ── V3: 보고된 세그먼트 수치와 대조 ─────────────────────────────────
def test_actual_share_confirms_core():
    v = verify_assignment(_asg(), PACK, SectorMatrix(), "ELEC_EQUIP")
    assert v.actual_share == pytest.approx(0.72)
    assert v.role == "core" and v.status == "verified"
    assert v.share_evidence == "confirmed"


def test_claim_gap_is_recorded_but_does_not_gate():
    """LLM 이 비중을 부풀리면 보고 수치가 잡는다 — 다만 **기록만** 한다.

    강등 근거로 쓰면 LLM 이 뱉은 연속값이 판정을 흔든다. 실제로 등급 체계에서
    이게 하드 플래그였을 때 재실행 tier 일치율이 84% 였다. 숫자는 리포트에
    싣고 판단은 사람이 한다.
    """
    v = verify_assignment(_asg(segment_name="배전", revenue_share_claim=0.80),
                          PACK, SectorMatrix(), "ELEC_EQUIP")
    assert v.actual_share == pytest.approx(0.28)
    assert v.claimed_share == pytest.approx(0.80)
    assert any(f.startswith("share_claim_gap") for f in v.flags)
    assert v.status == "verified" and v.role == "core"   # 게이트하지 않는다


def test_actual_share_below_floor_downgrades_core():
    """매출 2% 신사업을 core 라고 주장해도 실제 수치가 강등시킨다."""
    pack = {**PACK, "segments": {"complete": True, "corroborated": True,
                                 "source_line": "x",
                                 "segments": [{"name": "중전기기", "share": 0.02}]}}
    v = verify_assignment(_asg(revenue_share_claim=0.02), pack,
                          SectorMatrix(), "ELEC_EQUIP")
    assert v.role == "peripheral"
    assert any(f.startswith("role_downgraded_by_actual") for f in v.flags)


def test_nonexistent_segment_name_flagged():
    v = verify_assignment(_asg(segment_name="존재하지않는부문"), PACK,
                          SectorMatrix(), "ELEC_EQUIP")
    assert any(f.startswith("segment_not_found") for f in v.flags)
    assert v.share_evidence == "not_found"   # 대조할 목록이 있었는데 없었다
    assert v.status == "verified"            # 폐기 사유는 인용 환각뿐이다


# ── 판정 경계 ────────────────────────────────────────────────────────
def test_only_two_reject_reasons_exist():
    """폐기는 '지어냈다'와 '사전에 없다' 둘뿐이다. 나머지는 전부 표시다."""
    src = (Path(__file__).resolve().parents[1]
           / "src/pipeline/verify/layers.py").read_text(encoding="utf-8")
    assert set(re.findall(r'REJECTED, "(\w+)"', src)) == {
        "unknown_theme", "hallucinated_citation"}


def test_no_tier_machinery_remains():
    """등급을 지웠으면 흔적도 남기지 않는다 — 반쯤 남으면 둘 다 참조된다."""
    src = (Path(__file__).resolve().parents[1]
           / "src/pipeline/verify/layers.py").read_text(encoding="utf-8")
    for gone in ("def assign_tier", "def hard_flag_count", "MIN_CONFIDENCE_TIER_A",
                 "HARD_FLAGS", "UNVERIFIABLE_FLAGS"):
        assert gone not in src, f"{gone} 가 남아 있다"


def test_confidence_no_longer_decides_anything():
    """A/B 를 가르던 유일한 기준이 LLM 자기신고 confidence 였다.
    모델이 매긴 소수점이 등급을 정하면 그건 검증이 아니다."""
    lo = verify_assignment(_asg(confidence=0.10), PACK, SectorMatrix(), "ELEC_EQUIP")
    hi = verify_assignment(_asg(confidence=0.99), PACK, SectorMatrix(), "ELEC_EQUIP")
    assert lo.status == hi.status and lo.role == hi.role
    assert lo.share_evidence == hi.share_evidence


def test_core_without_share_evidence_is_not_a_failure():
    """비중을 '말하지 않은 것'은 배정이 틀렸다는 증거가 아니다.
    골든셋 실측에서 이 플래그가 붙은 5건이 전부 정답이었다 — 판별력 0."""
    v = verify_assignment(_asg(revenue_share_claim=0.0, segment_name=None),
                          PACK_NO_SEG, SectorMatrix(), "ELEC_EQUIP")
    assert "core_without_share_evidence" in v.flags
    assert v.status == "verified" and v.role == "core"
    assert v.share_evidence == "not_claimed"


def test_unavailable_segments_do_not_downgrade():
    """확인 불가를 실패로 취급하면 세그먼트를 공시 안 하는 회사가 벌을 받는다."""
    v = verify_assignment(_asg(segment_name=None), PACK_NO_SEG,
                          SectorMatrix(), "ELEC_EQUIP")
    assert "segment_data_unavailable" in v.flags
    assert v.status == "verified" and v.role == "core"
    assert v.share_evidence == "unavailable"


def test_segment_name_fuzzy_match():
    """LLM 이 '중전기기 부문'이라 써도 원문 '중전기기'를 찾아야 한다."""
    v = verify_assignment(_asg(segment_name="중전기기 부문"), PACK,
                          SectorMatrix(), "ELEC_EQUIP")
    assert v.actual_share == pytest.approx(0.72)



def test_uncorroborated_segments_not_used_for_downgrade():
    """뒷받침 없는 비율표(table_pct)로 core 를 깎으면 검증이 아니라 추측이 된다.
    확인 불가로 처리해 tier B 에 두되, 강등은 하지 않는다."""
    pack = {**PACK, "segments": {"complete": True, "corroborated": False,
                                 "method": "table_pct", "source_line": "x",
                                 "segments": [{"name": "중전기기", "share": 0.02}]}}
    v = verify_assignment(_asg(), pack, SectorMatrix(), "ELEC_EQUIP")
    assert v.actual_share is None
    assert v.role == "core"                      # 강등하지 않는다
    assert "segment_data_unverified" in v.flags   # '없음'이 아니라 '뒷받침 없음'
    assert v.share_evidence == "unverified"      # 그러나 '확인됨'도 아니다


# ── 태그 캐시 무효화 ─────────────────────────────────────────────────
def test_cache_key_changes_when_theme_definition_changes():
    """택소노미를 고치면 캐시가 반드시 무효화돼야 한다.

    키를 taxonomy['version'] 같은 수기 문자열에 걸면, 정의만 고치고 버전을
    안 올렸을 때 재실행이 조용히 옛 결과를 돌려준다. 실제로 shipping_freight
    제외 조항을 추가하고 재실행했는데 62종목 결과가 한 건도 안 바뀌었다.
    """
    import copy
    import yaml
    from pathlib import Path
    from pipeline.llm.cascade import cache_key, system_fingerprint

    repo = Path(__file__).resolve().parents[1]
    tax = yaml.safe_load(
        (repo / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))
    edited = copy.deepcopy(tax)
    edited["themes"][0]["exclusion"] = "테스트용으로 바꾼 제외 조항"
    # version 은 일부러 그대로 둔다 — 그래도 키가 달라져야 한다.
    assert edited["version"] == tax["version"]

    fp_a, fp_b = system_fingerprint(tax), system_fingerprint(edited)
    assert fp_a != fp_b, "테마 정의를 바꿨는데 시스템 프롬프트 지문이 그대로다"
    assert (cache_key("m", fp_a, "pack", ["t"])
            != cache_key("m", fp_b, "pack", ["t"]))


def test_cache_key_is_stable_for_identical_input():
    """같은 입력이면 같은 키. 아니면 캐시가 무용지물이고 매번 재과금된다."""
    import yaml
    from pathlib import Path
    from pipeline.llm.cascade import cache_key, system_fingerprint
    repo = Path(__file__).resolve().parents[1]
    tax = yaml.safe_load(
        (repo / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))
    fp = system_fingerprint(tax)
    assert (cache_key("m", fp, "pack", ["b", "a"])
            == cache_key("m", fp, "pack", ["a", "b"]))       # 후보 순서 무관
