"""일별 운영 스크립트.

스케줄러가 도는 코드라 사람이 안 보는 시간에 실패한다. 조용한 오작동을 막는
장치(거래일 판정, 부분 적재 감지, 실패 시 비정상 종료)를 고정한다.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PY = REPO / "scripts" / "daily_prices.py"
PS1 = REPO / "scripts" / "daily_prices.ps1"


def _mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("daily_prices", PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_script_parses():
    ast.parse(PY.read_text(encoding="utf-8"))


def test_partial_load_is_not_treated_as_done():
    """행이 있다는 것과 다 받았다는 것은 다르다. 4종목만 넣어둔 상태를
    '완료'로 읽으면 그날 시세가 영영 안 채워진다."""
    m = _mod()
    assert m.MIN_COVERAGE >= 0.5, "커버리지 기준이 없으면 부분 적재를 걸러낼 수 없다"
    src = PY.read_text(encoding="utf-8")
    assert "got / want >= MIN_COVERAGE" in src, "건너뛰기 조건이 커버리지 기준을 안 쓴다"


def test_as_of_is_a_trading_day_not_today():
    """build_price_table 이 date=as_of 로 저장하므로, 토요일에 오늘 날짜로
    돌리면 금요일 종가가 토요일 행이 된다. 휴장일 유령 행이 생긴다."""
    src = PY.read_text(encoding="utf-8")
    assert "last_trading_day" in src
    assert "date.today()" in src and "else last_trading_day" in src


def test_trading_day_uses_probe_not_holiday_calendar():
    """공휴일 달력은 매년 갱신해야 하고 임시휴장을 못 잡는다."""
    m = _mod()
    assert m.PROBE_TICKER and m.PROBE_LOOKBACK_DAYS >= 7
    assert "holiday" not in PY.read_text(encoding="utf-8").lower()


def test_failure_exits_nonzero():
    """조용히 성공으로 끝내면 스케줄러가 실패를 기록하지 못하고,
    다음 날 이상을 눈치채지 못한다."""
    src = PY.read_text(encoding="utf-8")
    assert 'rec["status"] = "failed"' in src
    assert "return 1" in src


def test_run_is_logged_with_coverage():
    """로그가 없으면 '어제 몇 종목 받았나'에 답할 수 없다."""
    m = _mod()
    assert m.LOG.suffix == ".jsonl"          # 한 줄 = 한 실행, grep 가능
    src = PY.read_text(encoding="utf-8")
    for k in ('"coverage"', '"elapsed_s"', '"as_of"'):
        assert k in src, f"로그에 {k} 가 없다"


# ── PowerShell 래퍼 ──────────────────────────────────────────────────
@pytest.mark.skipif(not PS1.exists(), reason="래퍼 없음")
def test_wrapper_sets_both_encodings():
    """PYTHONIOENCODING 만 설정하면 PowerShell 이 UTF-8 바이트를 cp949 로
    오독해 로그 파일에 깨진 글자가 남는다(실측). 양쪽 다 필요하다."""
    t = PS1.read_text(encoding="utf-8")
    assert "PYTHONIOENCODING" in t
    assert "OutputEncoding" in t


@pytest.mark.skipif(not PS1.exists(), reason="래퍼 없음")
def test_wrapper_propagates_exit_code():
    t = PS1.read_text(encoding="utf-8")
    assert "LASTEXITCODE" in t and "exit $rc" in t


@pytest.mark.skipif(not PS1.exists(), reason="래퍼 없음")
def test_wrapper_does_not_duplicate_trading_day_logic():
    """요일 조건을 스케줄러와 스크립트 양쪽에 두면 임시휴장 때 어긋난다."""
    t = PS1.read_text(encoding="utf-8")
    assert "-DaysOfWeek" not in t


# ── 실제 로그 ────────────────────────────────────────────────────────
def test_logged_runs_are_readable():
    m = _mod()
    if not m.LOG.exists():
        pytest.skip("실행 기록 없음")
    import json
    for line in m.LOG.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        assert rec.get("status") in ("ok", "skipped", "failed",
                                     "derive_failed", "probe_failed")
