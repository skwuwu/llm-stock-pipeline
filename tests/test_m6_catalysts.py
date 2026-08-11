"""촉매 — 공시된 사실만. LLM 판단이 들어갈 자리가 없어야 한다.

이 모듈의 존재 이유는 직전 리팩터에서 배운 것이다: LLM 이 매긴 연속값이
판정을 정하면 재실행마다 순위가 바뀐다(실측 판정 일치율 81%).
촉매는 처음부터 그 문제가 생기지 않게 설계했다 — kind 는 공시유형이,
magnitude 는 공시 숫자가, expires_at 은 규칙이 정한다.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.catalysts.build import (SELF_COMPUTED, amendment_index, classify,
                                      load_catalog)
from pipeline.catalysts.extract import (STRUCTURED, _num, from_contract_document,
                                        normalize, resolve_expiry)
from pipeline.ingest.dart_disclosure import normalize_report_name
from pipeline.ingest.dart_document import to_plain_text

REPO = Path(__file__).resolve().parents[1]
CATALOG = REPO / "configs/catalysts/catalyst_v1.yaml"


# ── 카탈로그가 닫혀 있는가 ───────────────────────────────────────────
def test_catalog_is_a_closed_list():
    """자유 텍스트 kind 를 허용하면 '촉매인 것 같은 것'이 슬금슬금 들어온다."""
    specs, cfg = load_catalog()
    ids = [s.id for s in specs]
    assert len(ids) == len(set(ids)), f"중복 id {ids}"
    assert all(re.fullmatch(r"[CX]\d+", i) for i in ids), ids
    assert {s.polarity for s in specs} == {"positive", "negative"}


def test_negative_catalysts_exist_and_are_enabled():
    """촉매만 찾는 시스템은 낙관 편향을 갖는다. 실측: 주요사항보고 535건 중
    222건(41%)이 유상증자·CB 였다 — 이걸 빼면 가장 흔한 신호를 놓친다."""
    specs, _ = load_catalog()
    neg = [s for s in specs if s.polarity == "negative"]
    assert neg, "역촉매가 하나도 없다"
    assert any(s.enabled for s in neg), "역촉매가 전부 꺼져 있다"


def test_every_catalyst_has_an_anchor_and_expiry():
    """rcept_no 없는 촉매는 존재할 수 없고, 시효 없는 촉매는 영원히 남는다."""
    specs, _ = load_catalog()
    for s in specs:
        assert s.pblntf_ty, f"{s.id}: 앵커 공시유형이 없다"
        assert s.pattern.pattern != "(?!)", f"{s.id}: 패턴이 비었다"
        assert s.expires_days and s.expires_days > 0, f"{s.id}: 시효가 없다"


def test_exclusions_carry_a_reason():
    """'왜 제외인지' 를 안 적으면 다음 사람이 그냥 넣는다."""
    cfg = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    for e in cfg["excluded"]:
        assert e.get("why"), f"{e['name']}: 제외 사유가 없다"


def test_confidence_checks_are_all_deterministic():
    """LLM 자기신고가 하나라도 섞이면 랭킹이 재현되지 않는다."""
    cfg = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    ids = {c["id"] for c in cfg["confidence_checks"]}
    assert ids == {"grounded", "not_amended", "not_expired",
                   "material", "no_reversal", "numbers_agree"}
    blob = " ".join(c["what"] for c in cfg["confidence_checks"])
    for banned in ("confidence", "LLM", "모델"):
        assert banned not in blob, f"체크 설명에 {banned} 가 들어 있다"


# ── 숫자 파싱 ────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,want", [
    ("53,500,000,000", 5.35e10), ("145.42", 145.42), ("0", 0.0),
    ("-", None), ("", None), ("―", None), (None, None),
])
def test_num_treats_dash_as_missing_not_zero(raw, want):
    """'-' 를 0 으로 읽으면 '금액 미기재'와 '금액 0원'이 같아진다."""
    got = _num(raw)
    assert got == want if want is not None else got is None


def test_normalize_refuses_bad_denominator():
    assert normalize(100.0, 0) is None
    assert normalize(100.0, -5) is None
    assert normalize(100.0, None) is None
    assert normalize(None, 100.0) is None
    assert normalize(50.0, 200.0) == pytest.approx(0.25)


# ── 시효 ─────────────────────────────────────────────────────────────
def test_document_end_date_beats_default_days():
    """공시가 계약종료일을 주면 그것이 진실이다. 기본 일수는 없을 때만."""
    d = date(2026, 8, 6)
    assert resolve_expiry(d, 365, from_document=date(2029, 11, 30)) == date(2029, 11, 30)
    assert resolve_expiry(d, 180, from_document=None) == date(2027, 2, 2)
    assert resolve_expiry(d, None, None) is None


# ── 정정공시 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,key,amend", [
    ("[기재정정]주요사항보고서(자기주식취득결정)", "주요사항보고서(자기주식취득결정)", True),
    ("주요사항보고서(유상증자결정)", "주요사항보고서(유상증자결정)", False),
    ("[첨부정정]단일판매ㆍ공급계약체결", "단일판매ㆍ공급계약체결", True),
])
def test_amendment_prefix_stripped_but_flag_kept(raw, key, amend):
    """접두어를 떼어 패턴 매칭하되 **정정이라는 사실은 버리지 않는다** —
    촉매 검증의 첫 체크가 '이 건이 뒤집혔나' 이기 때문이다."""
    assert normalize_report_name(raw) == (key, amend)


def test_amendment_index_pairs_ticker_and_report():
    d = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "report_key": ["주요사항보고서(유상증자결정)"] * 3,
        "is_amendment": [True, False, False],
    })
    assert amendment_index(d) == {("A", "주요사항보고서(유상증자결정)")}


# ── 분류 ─────────────────────────────────────────────────────────────
def _disc(**kw):
    base = {"ticker": "000001", "report_key": "주요사항보고서(자기주식취득결정)",
            "pblntf_ty": "B", "is_amendment": False, "rcept_no": "1",
            "rcept_dt": date(2026, 8, 1)}
    return pd.DataFrame([{**base, **kw}])


def test_classify_matches_by_type_and_pattern():
    specs, _ = load_catalog()
    assert classify(_disc(), specs)["kind"].tolist() == ["C1"]
    # 같은 이름이라도 공시유형이 다르면 매칭되지 않는다
    assert classify(_disc(pblntf_ty="I"), specs).empty


def test_excluded_report_types_never_classify():
    """제외 목록의 것들이 실수로 촉매가 되면 안 된다."""
    specs, _ = load_catalog()
    for nm in ("기업설명회(IR)개최", "주주총회소집결의", "감사보고서제출",
               "최대주주등소유주식변동신고서", "주주명부폐쇄기간또는기준일설정"):
        for ty in ("B", "I"):
            assert classify(_disc(report_key=nm, pblntf_ty=ty), specs).empty, nm


def test_bonus_issue_is_not_dilution():
    """무상증자는 주식수가 늘어도 지분율이 그대로다 — 희석이 아니다."""
    specs, _ = load_catalog()
    assert classify(_disc(report_key="주요사항보고서(무상증자결정)"), specs).empty


# ── C2 문서 파싱 ─────────────────────────────────────────────────────
C2_DOC = """<style>.xforms td { padding:0 }</style>
<TABLE><TR><TD>계약금액 총액(원)</TD><TD>53,500,000,000</TD></TR>
<TR><TD>최근 매출액(원)</TD><TD>36,790,989,927</TD></TR>
<TR><TD>매출액 대비(%)</TD><TD>145.42</TD></TR>
<TR><TD>시작일</TD><TD>2026-08-06</TD><TD>종료일</TD><TD>2029-11-30</TD></TR></TABLE>"""


def test_contract_parsed_through_table_separators():
    """to_plain_text 가 셀을 ' | ' 로 잇는다. 라벨과 값 사이 구분자를
    건너뛰지 못하면 전부 None 이 된다(실제로 그랬다)."""
    m = from_contract_document(to_plain_text(C2_DOC))
    assert m is not None
    assert m.value == pytest.approx(1.4542)
    assert m.expires_at == date(2029, 11, 30)
    assert m.fields["계약금액총액"] == pytest.approx(5.35e10)


def test_style_block_is_stripped_before_parsing():
    """CSS 가 남으면 그 안의 숫자가 잡히고, LLM 입력·V1 대조에도 섞인다."""
    t = to_plain_text(C2_DOC)
    assert "padding" not in t and "xforms" not in t


# ── 자체 계산 경로 ───────────────────────────────────────────────────
def test_self_computed_uses_delta_not_growth_rate():
    """**증가율이 아니라 증가액이다.**

    비율은 기저가 작으면 폭주한다 — 전년 동기 영업이익 1억 → 30억이면
    2,900% 다. 실측에서 실적 서프라이즈 상위가 전부 그런 종목이었다
    (해성디에스 2,995% / 펄어비스 2,597%). 증가액을 시총으로 나누면
    작은 기저에서 나온 큰 % 가 자동으로 작은 값이 된다.
    """
    assert SELF_COMPUTED == {"C3": "op_delta_q"}
    assert "growth" not in SELF_COMPUTED["C3"], "증가율로 되돌아갔다"
    assert "C3" not in STRUCTURED


def test_ratio_valued_catalysts_are_not_divided_again():
    """추출값이 이미 비율인데 denominator 가 있으면 또 나눠져 0 이 된다.

    실측: C2 에 denominator: revenue_ttm 이 남아 있어 전 건이 0% 로 찍혔다.
    """
    specs = {s.id: s for s in load_catalog()[0]}
    assert specs["C2"].mag_denominator is None, (
        "C2 는 공시가 '매출액 대비(%)' 를 이미 계산해 준다 — 다시 나누면 안 된다")
    # 반대로 절대금액을 쓰는 촉매는 분모가 반드시 있어야 한다
    for k in ("C1", "C3", "X1"):
        assert specs[k].mag_denominator, f"{k}: 절대금액인데 분모가 없다"


# ── 산출물 ───────────────────────────────────────────────────────────
def test_emitted_catalysts_are_wellformed():
    p = REPO / "data/screens/theme_hunt/2026-08-06/catalysts.parquet"
    if not p.exists():
        pytest.skip("촉매 산출물 없음")
    df = pd.read_parquet(p)
    specs, _ = load_catalog()
    by = {s.id: s for s in specs}
    assert set(df["kind"]) <= set(by), "카탈로그에 없는 kind 가 있다"
    assert df["rcept_no"].notna().all(), "앵커 없는 촉매가 있다"
    assert df["confidence"].between(0, 6).all()
    # 크기를 **모르는** 것은 버린다(순위를 못 매기므로). 작은 것은 남긴다.
    for k, g in df.groupby("kind"):
        if by[k].mag_min is not None:
            assert g["magnitude"].notna().all(), f"{k}: 크기 미상이 섞였다"


def test_small_catalysts_are_flagged_not_dropped():
    """시총 0.5% 자사주도 일어난 사실이다. 버리면 '작았다'와 '없었다'가 같아진다.

    magnitude.min 은 필터가 아니라 신뢰도 체크(material)의 임계다.
    """
    p = REPO / "data/screens/theme_hunt/2026-08-06/catalysts.parquet"
    if not p.exists():
        pytest.skip("촉매 산출물 없음")
    df = pd.read_parquet(p)
    by = {s.id: s for s in load_catalog()[0]}
    small = df[df.apply(lambda r: by[r["kind"]].mag_min is not None
                        and pd.notna(r["magnitude"])
                        and r["magnitude"] < by[r["kind"]].mag_min, axis=1)]
    if small.empty:
        pytest.skip("임계 미만 촉매가 없다")
    assert not small["chk_material"].any(), "임계 미만인데 material 이 True 다"
    assert (small["confidence"] < 6).all(), "임계 미만인데 만점이다"


def test_expired_catalysts_are_marked_not_dropped():
    """만료를 조용히 버리면 '없었다'와 구분되지 않는다. 체크로 남긴다."""
    p = REPO / "data/screens/theme_hunt/2026-08-06/catalysts.parquet"
    if not p.exists():
        pytest.skip("촉매 산출물 없음")
    df = pd.read_parquet(p)
    assert "chk_not_expired" in df.columns


# ── 다이제스트 연결 ──────────────────────────────────────────────────
def _digest(cat_rows, **kw):
    from datetime import date as _d

    from pipeline.report.digest import render
    surv = pd.DataFrame([{"ticker": "000001", "name": "가나전기",
                          "sector_code": "ELEC_EQUIP", "per": 8.0, "pbr": 0.9,
                          "fcf_yield": 0.1, "risk_groups": 0}])
    verd = pd.DataFrame([{"ticker": "000001", "theme_id": "power_cable_grid",
                          "role": "core", "status": "verified", "flags": "",
                          "reject_reason": None, "confidence": 0.9,
                          "share_evidence": "confirmed", "actual_share": 0.8,
                          "claimed_share": 0.8, "rationale": "전선 제조",
                          "evidence_quote": "전선을 제조한다"}])
    tax = {"themes": [{"id": "power_cable_grid", "name_ko": "전력망·전선"}]}
    return render(_d(2026, 8, 6), surv, verd, tax, {}, {},
                  {"status": {"verified": 1}, "verified_core": 1, "assignments": 1,
                   "hallucinated_citation_rate": 0.0},
                  catalysts=pd.DataFrame(cat_rows), **kw)


def _cat(**kw):
    base = {"ticker": "000001", "kind": "C1", "name": "자사주 취득·소각",
            "polarity": "positive", "rcept_no": "20260801000001",
            "occurred_at": date(2026, 8, 1), "expires_at": date(2027, 1, 28),
            "magnitude": 0.05, "magnitude_basis": "tsstkAqDecsn:aqpln_prc_ostk",
            "report_key": "주요사항보고서(자기주식취득결정)", "confidence": 6,
            "chk_grounded": True, "chk_not_amended": True, "chk_not_expired": True,
            "chk_material": True, "chk_no_reversal": True, "chk_numbers_agree": True}
    return {**base, **kw}


def test_digest_works_without_catalysts():
    """촉매를 안 돌린 스크린도 다이제스트는 나와야 한다."""
    md = _digest([])
    assert "전력망·전선" in md and "역촉매" not in md


def test_counter_catalysts_come_before_catalysts():
    """나쁜 소식이 스크롤 아래에 있으면 안 읽힌다.
    실측: 주요사항보고의 41% 가 유상증자·CB 였다."""
    md = _digest([_cat(), _cat(kind="X1", polarity="negative",
                               name="지분 희석", rcept_no="2",
                               report_key="주요사항보고서(유상증자결정)")])
    assert md.index("## 역촉매") < md.index("## 촉매")


def test_failed_checks_are_named_not_just_counted():
    """'5/6' 만으로는 무엇이 빠졌는지 모른다. 규모 미달과 정정있음은 다르다."""
    md = _digest([_cat(confidence=4, chk_material=False, chk_not_amended=False)])
    assert "규모미달" in md and "정정있음" in md


def test_expired_catalysts_go_to_appendix_not_the_void():
    """조용히 버리면 '촉매가 없었다'와 '만료됐다'가 같아진다."""
    md = _digest([_cat(chk_not_expired=False, confidence=5,
                       expires_at=date(2026, 1, 1))])
    assert "부록 C" in md
    body = md.split("## 부록 C")[0]
    assert "20260801000001" not in body


def test_dilution_kinds_are_distinguishable():
    """유상증자와 전환사채가 둘 다 '지분 희석' 이라 표에서 구분이 안 됐다.
    실측: 큐로셀이 같은 날 각 9.2% 씩 두 건 — 합산 18.4% 인데 중복으로 읽혔다."""
    md = _digest([
        _cat(kind="X1", polarity="negative", name="지분 희석", rcept_no="1",
             report_key="주요사항보고서(유상증자결정)", magnitude=0.092),
        _cat(kind="X1", polarity="negative", name="지분 희석", rcept_no="2",
             report_key="주요사항보고서(전환사채권발행결정)", magnitude=0.092),
    ])
    assert "유상증자결정" in md and "전환사채권발행결정" in md


def test_theme_table_carries_a_catalyst_column():
    md = _digest([_cat()])
    hdr = next(l for l in md.splitlines() if l.startswith("| 종목 | 촉매"))
    assert "촉매" in hdr
