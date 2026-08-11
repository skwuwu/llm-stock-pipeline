"""생존편향 — 없앨 수 없으면 재기라도 해야 한다.

KIND 상장법인목록은 **현재 상장사만** 준다. 폐지된 회사는 목록에서 사라지므로
security_master 에도 없고, 과거 as_of 로 스크린을 돌리면 '그때 있었지만 지금
없는' 회사를 못 본다. 살아남은 것만 남아 성과가 부풀려진다.

실측(2023-01-01): 그때 상장 2,552종목 중 **216종목**이 이후 폐지됐다
(failed 101 / merged 66 / dissolved 37 / voluntary 6 / other 6).
그 216 이 유니버스에 없다는 사실을 조용히 두면 안 된다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from pipeline.ingest.delisting import (COMPENSATED, DISSOLVED, FAILED, MERGED,
                                       OTHER, VOLUNTARY, classify_reason,
                                       coverage_report)

REPO = Path(__file__).resolve().parents[1]
DB = REPO / "data/pit.duckdb"


# ── 사유 분류 ────────────────────────────────────────────────────────
# 이게 틀리면 편향을 고치려다 **반대 방향** 편향을 만든다.
@pytest.mark.parametrize("reason,want", [
    ("피흡수합병", MERGED),
    ("피흡수합병(스팩소멸합병)", MERGED),
    ("지주회사(최대주주등)의 완전자회사화 등", MERGED),
    ("타법인의 완전자회사로 편입", MERGED),
    ("주식의 포괄적 교환", MERGED),
    ("상장폐지 신청", VOLUNTARY),
    ("신청에 의한 상장폐지", VOLUNTARY),
    ("상장폐지 신청('25.11.06)", VOLUNTARY),
    ("존속기간 만료", DISSOLVED),
    ("감사의견 거절(감사범위 제한)", FAILED),
    ("기업의 계속성 및 경영의 투명성 등을 종합적으로 고려하여 상장폐지기준에 해당한다고 결정", FAILED),
    ("해산 사유 발생", FAILED),
    ("자본전액잠식", FAILED),
    ("시가총액 미달", FAILED),
    ("분산요건 미달", FAILED),
    ("사업보고서 법정제출기한('21.3.31限)내 미제출", FAILED),
    ("지정자문인 선임계약 해지 후 30일 이내 미체결(2020.2.13 限)", FAILED),
])
def test_reason_classification(reason, want):
    assert classify_reason(reason) == want, reason


def test_merger_is_not_a_total_loss():
    """더존비즈온(지주회사 완전자회사화)을 전손으로 처리하면
    편향을 고치려다 새 편향을 만든다."""
    assert classify_reason("지주회사(최대주주등)의 완전자회사화 등",
                           "더존비즈온") in COMPENSATED
    assert classify_reason("기업의 계속성 및 경영의 투명성 등을 종합적으로 고려하여 "
                           "상장폐지기준에 해당한다고 결정", "스타코링크") == FAILED


def test_spac_dissolution_read_by_name_not_reason():
    """스팩은 사유가 '심사청구서 미제출'이라 실패처럼 보인다."""
    r = "상장예비심사신청서 미제출 등으로 관리종목으로 지정된 날부터 1개월 이내 동 사유 미해소"
    assert classify_reason(r, "하나30호스팩") == DISSOLVED
    assert classify_reason(r, "IBKS제23호스팩") == DISSOLVED


def test_unknown_reason_is_not_pushed_into_failed():
    """모르는 것을 전손으로 세면 백테스트가 반대 방향으로 틀린다."""
    assert classify_reason("듣도 보도 못한 사유") == OTHER
    assert classify_reason("") == OTHER
    assert classify_reason(None) == OTHER
    assert OTHER not in COMPENSATED


def test_missing_reason_does_not_crash():
    """FDR 은 사유를 NaN(float) 으로 보내는 행이 있다."""
    import numpy as np
    assert classify_reason(np.nan) == OTHER
    assert classify_reason(np.nan, "무슨스팩") == DISSOLVED


def test_compensated_set_excludes_failure():
    assert FAILED not in COMPENSATED
    assert {MERGED, VOLUNTARY, DISSOLVED} == set(COMPENSATED)


# ── PIT 유니버스 ─────────────────────────────────────────────────────
def _store():
    if not DB.exists():
        pytest.skip("스토어 없음")
    from pipeline.store.pit import PitStore
    return PitStore(DB)


def _has_delistings(s) -> bool:
    return bool(s.con.execute("SELECT count(*) FROM delistings").fetchone()[0])


def test_listed_asof_includes_companies_delisted_since():
    """**이게 생존편향 수정의 핵심이다.** 과거 시점 유니버스는
    '지금 상장 중'이 아니라 '그때 상장 중'이어야 한다."""
    s = _store()
    try:
        if not _has_delistings(s):
            pytest.skip("폐지 이력 없음")
        past = s.listed_asof(date(2023, 1, 1))
        now = s.listed_asof(date(2026, 8, 6))
        assert past["delisted"].any(), "과거 시점에 폐지 종목이 하나도 없다"
        assert not now["delisted"].any(), "현재 시점인데 폐지 종목이 섞였다"
        # 과거 유니버스가 현재보다 작을 수는 있으나(그 사이 신규상장),
        # 폐지분이 반드시 포함돼 있어야 한다
        assert len(past[past["delisted"]]) > 50, "폐지 반영이 너무 적다"
    finally:
        s.close()


def test_listed_asof_excludes_not_yet_listed():
    """상장 전 종목이 들어오면 룩어헤드다."""
    import pandas as pd
    s = _store()
    try:
        cut = date(2015, 1, 1)
        u = s.listed_asof(cut)
        # DuckDB 가 DATE 를 datetime64 로 돌려준다. 경계에서 맞춘다.
        ld = pd.to_datetime(u["listing_date"], errors="coerce")
        late = u[ld.notna() & (ld > pd.Timestamp(cut))]
        assert late.empty, f"상장 전 종목이 섞였다: {len(late)}"
    finally:
        s.close()


def test_survivorship_report_quantifies_the_gap():
    """숫자가 없으면 '편향이 있다'는 말은 아무 일도 하지 않는다."""
    s = _store()
    try:
        if not _has_delistings(s):
            pytest.skip("폐지 이력 없음")
        r = s.survivorship_report(date(2023, 1, 1))
        for k in ("listed_then", "delisted_since", "delisted_by_outcome",
                  "missing_financials", "coverage"):
            assert k in r, k
        assert r["delisted_since"] > 0
        assert r["listed_then"] == r["still_listed"] + r["delisted_since"]
        # 결과별 분해가 있어야 합병과 전손을 구분해 쓸 수 있다
        assert set(r["delisted_by_outcome"]) & {FAILED, MERGED}
    finally:
        s.close()


def test_current_asof_has_no_survivorship_gap():
    """지금 시점 스크린에는 편향이 없다 — 지금 상장된 것을 보는 게 맞다.
    편향은 **과거 시점**에서만 생긴다."""
    s = _store()
    try:
        if not _has_delistings(s):
            pytest.skip("폐지 이력 없음")
        r = s.survivorship_report(date(2026, 8, 6))
        assert r["delisted_since"] == 0
    finally:
        s.close()


# ── 조용히 넘어가지 않는가 ───────────────────────────────────────────
def test_screen_warns_when_universe_is_survivor_only():
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    i = src.index("def cmd_screen")
    body = src[i:src.index("\ndef ", i + 10)]
    assert "생존편향" in body, "screen 이 생존편향을 언급하지 않는다"
    assert "survivorship_report" in body
    # 폐지 이력 자체가 없을 때도 조용하면 안 된다
    assert "폐지 이력이 없어" in body


def test_coverage_report_surfaces_unclassified():
    """분류 불가를 감추면 실패로 새는지 알 수 없다."""
    import pandas as pd
    d = pd.DataFrame({
        "delisting_date": [date(2024, 1, 1)] * 3,
        "outcome": [FAILED, MERGED, OTHER],
    })
    r = coverage_report(d)
    assert r["unclassified"] == 1
    assert r["by_outcome"][FAILED] == 1
