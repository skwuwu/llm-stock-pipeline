"""M0 완료 기준 검증: 임의 과거 날짜로 룩어헤드 없는 PER/PBR 재현.

네트워크 없이 돈다. 픽스처는 고정 DART 응답이다.

시나리오
  2025-03-20  FY2024 사업보고서 접수 (지배주주 순이익 1,000억, 지배주주지분 1조)
  2025-05-15  2025 1분기 보고서 접수 (1분기 누적 순이익 300억)
  2025-08-01  FY2024 정정공시 (순이익 1,000억 → 900억)

기대
  as_of 2025-04-01 → FY2024 만 보인다. 1분기는 아직 존재하지 않는다.
  as_of 2025-06-01 → TTM = 1,000억 − 250억 + 300억 = 1,050억
  as_of 2025-09-01 → 정정 반영. TTM = 900억 − 250억 + 300억 = 950억
  as_of 2025-04-01 을 다시 물어도 여전히 1,000억 — 미래의 정정이 과거로 새지 않는다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pipeline.derive.metrics import build_metrics
from pipeline.normalize.kr import compute_ttm, normalize_financials
from pipeline.store.pit import PitStore

FIX = Path(__file__).parent / "fixtures"
TICKER = "005930"
MARKET_CAP = 1_200_000_000_000.0   # 1.2조


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture
def store(tmp_path) -> PitStore:
    s = PitStore(tmp_path / "pit.duckdb")

    for fname, rev in (
        ("dart_fy2024_cfs.json", None),
        ("dart_q1_2025_cfs.json", None),
        ("dart_fy2024_cfs_revised.json", "20250320000123"),
    ):
        df = normalize_financials(load(fname), ticker=TICKER,
                                  statement_basis="CFS", revision_of=rev)
        assert not df.empty, f"{fname} 파싱 결과가 비었다"
        s.append_facts(df)

    s.upsert_master(pd.DataFrame([{
        "ticker": TICKER, "corp_code": "00126380", "name": "테스트전자",
        "market": "KOSPI", "sector_code": "SEMI",
        "listing_date": date(1975, 6, 11), "delisting_date": None,
        "fiscal_month": 12, "is_preferred": False, "parent_ticker": None,
        "is_spac": False, "is_reit": False, "is_financial": False, "is_holding": False,
    }]))
    s.upsert_prices(pd.DataFrame([{
        "date": d, "ticker": TICKER, "close": 70_000.0,
        "shares_common": 15_000_000, "shares_preferred": 2_000_000,
        "treasury_shares": 500_000,
        "market_cap_common": 1_050_000_000_000.0, "market_cap_total": MARKET_CAP,
        "adtv_20d": 5_000_000_000.0,
    } for d in [date(2025, 4, 1), date(2025, 6, 1), date(2025, 9, 1)]]))
    yield s
    s.close()


# ── 룩어헤드 차단 ───────────────────────────────────────────────────
def test_future_filing_is_invisible(store):
    """1분기 보고서(5/15 접수)는 4/1 시점에 존재하지 않아야 한다."""
    facts = store.facts_asof(date(2025, 4, 1))
    assert set(facts["source_doc_id"]) == {"20250320000123"}

    later = store.facts_asof(date(2025, 6, 1))
    assert "20250515000456" in set(later["source_doc_id"])


def test_available_lag_is_applied(store):
    """접수 당일(3/20)에는 아직 쓸 수 없고, 익일부터 쓸 수 있다."""
    assert store.facts_asof(date(2025, 3, 20)).empty
    assert not store.facts_asof(date(2025, 3, 21)).empty


# ── TTM 복원 ────────────────────────────────────────────────────────
def test_annual_ttm(store):
    facts = store.facts_asof(date(2025, 4, 1))
    ttm = compute_ttm(facts, "NET_INCOME_CONTROLLING")
    row = ttm[ttm["ticker"] == TICKER].iloc[0]
    assert row["value"] == pytest.approx(100_000_000_000)
    assert row["ttm_reason"] == "annual"


def test_quarterly_ttm_rolls_correctly(store):
    """FY(y-1) − CUM(y-1,3m) + CUM(y,3m) = 1000 − 250 + 300 = 1050억.

    전년 동기 누적(250억)은 1분기 보고서의 frmtrm 비교치에서 온다 — 추가 호출 없음.
    """
    facts = store.facts_asof(date(2025, 6, 1))
    row = compute_ttm(facts, "NET_INCOME_CONTROLLING").iloc[0]
    assert row["value"] == pytest.approx(105_000_000_000)
    assert row["ttm_reason"] == "rolled:3m"


def test_ttm_incomplete_is_null_not_zero(tmp_path):
    """구성요소가 없으면 0으로 메우지 않고 값을 만들지 않는다."""
    s = PitStore(tmp_path / "p.duckdb")
    s.append_facts(normalize_financials(load("dart_q1_2025_cfs.json"),
                                        ticker=TICKER, statement_basis="CFS"))
    # 1분기 보고서만 있으면 전기 FY 가 없어 TTM 을 만들 수 없다
    row = compute_ttm(s.facts_asof(date(2025, 6, 1)), "NET_INCOME_CONTROLLING").iloc[0]
    assert row["value"] is None or pd.isna(row["value"])
    assert row["ttm_reason"].startswith("incomplete")
    s.close()


# ── 정정공시 ────────────────────────────────────────────────────────
def test_revision_does_not_leak_backwards(store):
    """8/1 정정이 4/1 질의에 새어들면 안 된다. 백테스트 정직성의 핵심."""
    before = compute_ttm(store.facts_asof(date(2025, 4, 1)), "NET_INCOME_CONTROLLING").iloc[0]
    assert before["value"] == pytest.approx(100_000_000_000)

    after = compute_ttm(store.facts_asof(date(2025, 9, 1)), "NET_INCOME_CONTROLLING").iloc[0]
    assert after["value"] == pytest.approx(95_000_000_000)   # 900 − 250 + 300


def test_revision_history_is_preserved(store):
    """원본을 UPDATE 하지 않고 append 했으므로 두 값이 모두 남아 있다."""
    h = store.revision_history(TICKER, "NET_INCOME_CONTROLLING")
    fy24 = h[h["fiscal_end_date"] == pd.Timestamp("2024-12-31")]
    assert sorted(fy24["value"]) == [90_000_000_000, 100_000_000_000]


# ── PER / PBR 재현 ──────────────────────────────────────────────────
@pytest.mark.parametrize("as_of, exp_per, exp_pbr", [
    (date(2025, 4, 1), 1_200 / 100,   1_200 / 1_000),   # FY2024 기준
    (date(2025, 6, 1), 1_200 / 105,   1_200 / 1_030),   # 1Q25 TTM 반영
    (date(2025, 9, 1), 1_200 / 95,    1_200 / 1_030),   # 정정 반영
])
def test_per_pbr_reproducible_at_past_dates(store, as_of, exp_per, exp_pbr):
    m = build_metrics(store.facts_asof(as_of), store.prices_asof(as_of),
                      store.master(), as_of)
    r = m[m["ticker"] == TICKER].iloc[0]
    assert r["per"] == pytest.approx(exp_per, rel=1e-6)
    assert r["pbr"] == pytest.approx(exp_pbr, rel=1e-6)


def test_market_cap_includes_preferred(store):
    """PER 분자는 보통주+우선주 시총. 보통주만 쓰면 우선주 있는 회사가 저평가로 보인다."""
    m = build_metrics(store.facts_asof(date(2025, 4, 1)),
                      store.prices_asof(date(2025, 4, 1)), store.master(), date(2025, 4, 1))
    assert m.iloc[0]["market_cap_used"] == pytest.approx(MARKET_CAP)


def test_fcf_uses_absolute_capex(store):
    """DART CAPEX 는 음수 표기. 부호를 그대로 더하면 FCF 가 부풀려진다."""
    m = build_metrics(store.facts_asof(date(2025, 4, 1)),
                      store.prices_asof(date(2025, 4, 1)), store.master(), date(2025, 4, 1))
    assert m.iloc[0]["fcf"] == pytest.approx(150_000_000_000 - 50_000_000_000)


# ── 품질 플래그 ─────────────────────────────────────────────────────
def test_flags_present(store):
    m = build_metrics(store.facts_asof(date(2025, 4, 1)),
                      store.prices_asof(date(2025, 4, 1)), store.master(), date(2025, 4, 1))
    r = m.iloc[0]
    assert not r["capital_impairment"]
    assert not r["negative_earnings"]
    assert not r["ttm_incomplete"]
    # 비지배지분 = 1150 − 1000 = 150억, 총자본의 13% → 임계(20%) 미만
    assert not r["minority_interest_large"]


def test_capital_impairment_blocks_negative_pbr(tmp_path):
    """자본잠식은 PBR 이 음수라 'PBR < 1' 을 그냥 통과한다. 플래그가 유일한 방어선."""
    payload = load("dart_fy2024_cfs.json")
    for row in payload["list"]:
        if row["account_id"] == "ifrs-full_EquityAttributableToOwnersOfParent":
            row["thstrm_amount"] = "-50,000,000,000"

    s = PitStore(tmp_path / "p.duckdb")
    s.append_facts(normalize_financials(payload, ticker=TICKER, statement_basis="CFS"))
    s.upsert_master(pd.DataFrame([{"ticker": TICKER, "name": "잠식전자", "market": "KOSPI",
                                   "sector_code": "SEMI", "fiscal_month": 12,
                                   "is_financial": False, "is_holding": False}]))
    s.upsert_prices(pd.DataFrame([{"date": date(2025, 4, 1), "ticker": TICKER,
                                   "close": 1000.0, "shares_common": 15_000_000,
                                   "treasury_shares": 0, "market_cap_common": MARKET_CAP,
                                   "market_cap_total": MARKET_CAP, "adtv_20d": 1e9}]))
    m = build_metrics(s.facts_asof(date(2025, 4, 1)), s.prices_asof(date(2025, 4, 1)),
                      s.master(), date(2025, 4, 1))
    r = m.iloc[0]
    assert r["pbr"] < 0                     # 필터 'PBR < 1' 은 이걸 통과시킨다
    assert bool(r["capital_impairment"])    # 하드 가드가 잡아야 한다
    s.close()


def test_idempotent_reingest(store):
    """같은 응답을 다시 넣어도 행이 늘지 않는다(멱등)."""
    n = store.append_facts(normalize_financials(load("dart_fy2024_cfs.json"),
                                                ticker=TICKER, statement_basis="CFS"))
    assert n == 0
