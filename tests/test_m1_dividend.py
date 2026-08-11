"""배당(DART alotMatter) 수집·파싱.

이 데이터는 PIT 규율이 특히 중요하다. 배당은 결산일이 아니라 **정기주총 뒤
사업보고서**로 확정 공시된다. 결산일 기준으로 알고 있다고 두면 3개월치 룩어헤드다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pipeline.ingest.dart_dividend import DividendFetchError, parse_dividend
from pipeline.screen.checks import CheckDataError, assert_gate_coverage, load_checks

RAW = {
    "status": "000",
    "list": [
        {"rcept_no": "20260318000857", "stlm_dt": "2025년 12월 31일",
         "se": "주당 현금배당금(원)", "stock_knd": "보통주",
         "thstrm": "1,300", "frmtrm": "1,050"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025년 12월 31일",
         "se": "주당 현금배당금(원)", "stock_knd": "우선주",
         "thstrm": "-", "frmtrm": "-"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025년 12월 31일",
         "se": "(연결)현금배당성향(%)", "stock_knd": "-", "thstrm": "25.13"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025년 12월 31일",
         "se": "현금배당수익률(%)", "stock_knd": "보통주", "thstrm": "3.54"},
    ],
}


def _by(rows, element):
    return next(r for r in rows if r["element"] == element)


# ── PIT ──────────────────────────────────────────────────────────────
def test_reported_at_is_report_filing_date_not_fiscal_end():
    """결산일(12/31)이 아니라 사업보고서 접수일(3/18)이 공시 시점이다.
    결산일로 두면 3개월치 룩어헤드가 생긴다."""
    rows = parse_dividend(RAW, "000850", lag_days=1)
    r = _by(rows, "DPS_CASH")
    assert r["fiscal_end_date"] == date(2025, 12, 31)
    assert r["reported_at"] == date(2026, 3, 18)
    assert r["available_at"] == date(2026, 3, 19)


def test_missing_rcept_no_raises():
    """접수번호가 없으면 PIT 시점을 정할 수 없다. 추정해서 넣지 않는다."""
    bad = {"status": "000", "list": [dict(RAW["list"][0], rcept_no="")]}
    with pytest.raises(DividendFetchError, match="rcept_no"):
        parse_dividend(bad, "000850")


# ── 파싱 ─────────────────────────────────────────────────────────────
def test_common_share_row_is_used():
    """스크린 대상은 보통주다. 우선주 행을 집으면 무배당으로 읽힌다."""
    assert _by(parse_dividend(RAW, "000850"), "DPS_CASH")["value"] == 1300.0


def test_payout_and_reported_yield_parsed():
    rows = parse_dividend(RAW, "000850")
    assert _by(rows, "PAYOUT_RATIO_PCT")["value"] == pytest.approx(25.13)
    assert _by(rows, "DIV_YIELD_REPORTED_PCT")["value"] == pytest.approx(3.54)


def test_reported_but_empty_means_zero_dividend():
    """보고는 됐는데 값이 '-' 면 무배당이다. 사실이므로 0 으로 확정한다."""
    raw = {"status": "000", "list": [dict(RAW["list"][0], thstrm="-")]}
    assert _by(parse_dividend(raw, "X"), "DPS_CASH")["value"] == 0.0


def test_no_response_means_unknown_not_zero():
    """응답 자체가 없으면 '모름' 이다. 0 으로 두면 미수집 종목이 무배당으로
    둔갑해 배당 스크린에서 조용히 탈락한다."""
    assert parse_dividend({"status": "013", "list": []}, "X") == []
    assert parse_dividend({"status": "000", "list": []}, "X") == []


def test_whitespace_variants_in_se_label():
    """'주당 현금배당금(원)' 의 공백 표기가 흔들려도 잡아야 한다."""
    raw = {"status": "000", "list": [dict(RAW["list"][0], se="주당현금배당금(원)")]}
    assert _by(parse_dividend(raw, "X"), "DPS_CASH")["value"] == 1300.0


# ── 게이트 커버리지 가드 ─────────────────────────────────────────────
def _gate(max_missing=0.05):
    return load_checks([{"id": "dy", "kind": "gate_filter", "enabled": True,
                         "metric": "div_yield", "direction": "higher_better",
                         "loose": 0.0, "tight": 0.04, "max_missing": max_missing}])


def test_gate_refuses_to_run_with_too_many_missing():
    """higher_better 게이트에서 NaN 은 탈락으로 떨어진다. 수집이 덜 된 채 켜면
    '데이터를 못 받은 종목' 이 '조건 미달' 과 구별되지 않고 사라진다."""
    df = pd.DataFrame({"div_yield": [0.05, None, None, None]})
    with pytest.raises(CheckDataError, match="결측률"):
        assert_gate_coverage(df, _gate())


def test_gate_runs_when_coverage_is_sufficient():
    df = pd.DataFrame({"div_yield": [0.05, 0.01, 0.0, 0.03] * 25 + [None]})
    assert_gate_coverage(df, _gate())          # 결측 1% < 허용 5%


def test_gate_coverage_can_be_explicitly_accepted():
    """감수하겠다면 설정으로 올린다 — 조용히 넘어가지는 않는다."""
    df = pd.DataFrame({"div_yield": [0.05, None]})
    with pytest.raises(CheckDataError):
        assert_gate_coverage(df, _gate())
    assert_gate_coverage(df, _gate(max_missing=0.60))


def test_disabled_gate_is_not_coverage_checked():
    specs = load_checks([{"id": "dy", "kind": "gate_filter", "enabled": False,
                          "metric": "div_yield", "direction": "higher_better",
                          "loose": 0.0, "tight": 0.04}])
    assert_gate_coverage(pd.DataFrame({"div_yield": [None, None]}), specs)


# ── 인증키 형식 검증 ─────────────────────────────────────────────────
def test_dart_key_format_validated():
    """잘린 키는 '등록되지 않은 인증키' 라는 원인 불명 에러로만 드러난다.
    실측: .env 에 37자로 저장돼 있었고 프로브를 돌려서야 알았다."""
    from pipeline.ingest.dart import DartKeyError, validate_dart_key
    assert validate_dart_key("a" * 40) == "a" * 40
    for bad in (None, "", "a" * 39, "a" * 41, "z" * 40):
        with pytest.raises(DartKeyError):
            validate_dart_key(bad)


def test_dart_key_error_never_echoes_the_key():
    """키 값이 예외 메시지에 실리면 로그·스택트레이스로 새어나간다."""
    from pipeline.ingest.dart import DartKeyError, validate_dart_key
    secret = "deadbeef" * 4 + "cafe"        # 36자 — 형식 위반
    try:
        validate_dart_key(secret)
    except DartKeyError as e:
        assert secret not in str(e)
        assert "36" in str(e)               # 길이는 알려준다


# ── 최대주주 ─────────────────────────────────────────────────────────
HOLDER_RAW = {
    "status": "000",
    "list": [
        {"rcept_no": "20260318000857", "stlm_dt": "2025-12-31", "stock_knd": "보통주",
         "nm": "권영열", "relate": "본인", "trmend_posesn_stock_qota_rt": "17.93"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025-12-31", "stock_knd": "보통주",
         "nm": "권형석", "relate": "친인척", "trmend_posesn_stock_qota_rt": "15.45"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025-12-31", "stock_knd": "보통주",
         "nm": "계", "relate": "", "trmend_posesn_stock_qota_rt": "48.78"},
        {"rcept_no": "20260318000857", "stlm_dt": "2025-12-31", "stock_knd": "우선주",
         "nm": "계", "relate": "", "trmend_posesn_stock_qota_rt": "-"},
    ],
}


def test_owner_stake_uses_related_party_total_not_individual():
    """지배구조 지표는 개인 최대주주가 아니라 특수관계인 합계다.
    개인만 보면 가족·계열사로 분산 보유한 회사가 취약해 보인다."""
    from pipeline.ingest.dart_holder import parse_holders
    rows = parse_holders(HOLDER_RAW, "000850")
    vals = {r["element"]: r["value"] for r in rows}
    assert vals["OWNER_STAKE_PCT"] == pytest.approx(48.78)
    assert vals["LARGEST_HOLDER_PCT"] == pytest.approx(17.93)


def test_owner_stake_falls_back_to_sum_without_total_row():
    from pipeline.ingest.dart_holder import parse_holders
    raw = {"status": "000", "list": HOLDER_RAW["list"][:2]}
    vals = {r["element"]: r["value"] for r in parse_holders(raw, "X")}
    assert vals["OWNER_STAKE_PCT"] == pytest.approx(33.38)


def test_holder_pit_anchor_is_filing_date():
    from pipeline.ingest.dart_holder import parse_holders
    r = parse_holders(HOLDER_RAW, "000850", lag_days=1)[0]
    assert r["reported_at"] == date(2026, 3, 18)
    assert r["available_at"] == date(2026, 3, 19)


def test_holder_no_data_is_unknown_not_zero():
    from pipeline.ingest.dart_holder import parse_holders
    assert parse_holders({"status": "013", "list": []}, "X") == []
