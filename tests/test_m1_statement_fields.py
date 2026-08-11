"""재무제표 구분별 '누적' 필드 규칙 회귀 테스트.

이 규칙은 실 DART 응답으로 확인한 것이고(주석에 근거 기재), 틀리면 분기 PER 이
조용히 3~4배 어긋난다. 파서를 만질 때 여기서 잡히도록 고정한다.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pipeline.normalize.kr import compute_ttm, normalize_financials


def _payload(rows: list[dict], reprt: str, year: str = "2025") -> dict:
    base = {"rcept_no": "20250515000001", "reprt_code": reprt, "bsns_year": year,
            "corp_code": "00000000", "currency": "KRW"}
    return {"status": "000", "list": [{**base, **r} for r in rows]}


def _val(df: pd.DataFrame, element: str, fiscal_end: str) -> float | None:
    m = df[(df.element == element) & (df.fiscal_end_date == date.fromisoformat(fiscal_end))]
    return None if m.empty else float(m.iloc[0]["value"])


# ── 손익계산서: 누적은 add 필드 ─────────────────────────────────────
def test_is_quarterly_uses_add_amount_not_three_month():
    """3분기 손익: thstrm_amount 는 3개월, thstrm_add_amount 가 9개월 누적.
    (삼성전자 3Q24 실측: 9.78조 vs 26.05조)"""
    df = normalize_financials(_payload([{
        "sj_div": "IS", "account_id": "ifrs-full_ProfitLossAttributableToOwnersOfParent",
        "thstrm_amount": "9,781,547,000,000",       # 3개월 — 쓰면 안 된다
        "thstrm_add_amount": "26,045,230,000,000",  # 9개월 누적 — 이걸 써야 한다
        "frmtrm_q_amount": "5,501,304,000,000",     # 전년 3개월 — 쓰면 안 된다
        "frmtrm_add_amount": "8,449,574,000,000",   # 전년 9개월 누적
    }], reprt="11014"), ticker="005930", statement_basis="CFS")

    assert _val(df, "NET_INCOME_CONTROLLING", "2025-09-30") == 26_045_230_000_000
    assert _val(df, "NET_INCOME_CONTROLLING", "2024-09-30") == 8_449_574_000_000


def test_is_quarterly_without_add_field_produces_nothing():
    """add 필드가 없으면 thstrm 은 3개월이라 누적 자리에 넣을 수 없다.
    3개월 값을 누적으로 오인하느니 값을 만들지 않는다."""
    df = normalize_financials(_payload([{
        "sj_div": "IS", "account_id": "ifrs-full_ProfitLossAttributableToOwnersOfParent",
        "thstrm_amount": "9,781,547,000,000",
        "frmtrm_q_amount": "5,501,304,000,000",
    }], reprt="11014"), ticker="005930", statement_basis="CFS")
    assert df.empty or "NET_INCOME_CONTROLLING" not in set(df["element"])


# ── 현금흐름표: 누적은 thstrm, 전기는 frmtrm_q ──────────────────────
def test_cf_quarterly_thstrm_is_cumulative():
    """현금흐름표는 본래 누적으로만 보고된다. frmtrm_q_amount 는 전년 동기 '누적'
    으로 손익계산서와 의미가 정반대다. (기아 2025 실측: 1Q 3.01 → FY 9.05조)"""
    df = normalize_financials(_payload([{
        "sj_div": "CF", "account_id": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        "thstrm_amount": "8,630,000,000,000",     # 9개월 누적
        "frmtrm_q_amount": "10,000,000,000,000",  # 전년 9개월 누적
    }], reprt="11014"), ticker="000270", statement_basis="CFS")

    assert _val(df, "CFO", "2025-09-30") == 8_630_000_000_000
    assert _val(df, "CFO", "2024-09-30") == 10_000_000_000_000


def test_cf_prior_enables_quarterly_ttm():
    """CF 전기 비교치가 있어야 분기 시점 FCF TTM 이 성립한다."""
    q = normalize_financials(_payload([{
        "sj_div": "CF", "account_id": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        "thstrm_amount": "2,610,740,000,000", "frmtrm_q_amount": "3,010,000,000,000",
    }], reprt="11013", year="2026"), ticker="000270", statement_basis="CFS")
    fy = normalize_financials(_payload([{
        "sj_div": "CF", "account_id": "ifrs-full_CashFlowsFromUsedInOperatingActivities",
        "thstrm_amount": "9,054,140,000,000", "frmtrm_amount": "12,564,370,000,000",
    }], reprt="11011", year="2025"), ticker="000270", statement_basis="CFS")

    ttm = compute_ttm(pd.concat([q, fy], ignore_index=True), "CFO").iloc[0]
    # FY2025 9.05 − 1Q25누적 3.01 + 1Q26누적 2.61 = 8.65조
    assert ttm["value"] == pytest.approx(8_654_880_000_000)
    assert ttm["ttm_reason"] == "rolled:3m"


# ── CAPEX 합산 ──────────────────────────────────────────────────────
def test_capex_sums_multiple_purchase_accounts():
    """CAPEX 는 단일 계정이 아니라 여러 취득 항목의 합.
    우선순위로 하나만 고르면 과소계상되어 FCF 가 부풀려진다."""
    df = normalize_financials(_payload([
        {"sj_div": "CF", "account_id": "ifrs-full_PurchaseOfPropertyPlantAndEquipment",
         "thstrm_amount": "-50,000,000,000"},
        {"sj_div": "CF", "account_id": "dart_PurchaseOfOtherPropertyPlantAndEquipment",
         "thstrm_amount": "-30,000,000,000"},
        {"sj_div": "CF", "account_id": "dart_PurchaseOfStructure",
         "thstrm_amount": "-20,000,000,000"},
    ], reprt="11011"), ticker="000270", statement_basis="CFS")

    assert _val(df, "CAPEX_PPE", "2025-12-31") == 100_000_000_000


def test_capex_excludes_disposals_and_depreciation():
    """처분(유입)과 감가상각 조정을 CAPEX 에 섞으면 FCF 가 조용히 틀린다."""
    df = normalize_financials(_payload([
        {"sj_div": "CF", "account_id": "ifrs-full_PurchaseOfPropertyPlantAndEquipment",
         "thstrm_amount": "-50,000,000,000"},
        {"sj_div": "CF",
         "account_id": "ifrs-full_ProceedsFromSalesOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
         "thstrm_amount": "40,000,000,000"},
        {"sj_div": "CF", "account_id": "ifrs-full_AdjustmentsForAmortisationExpense",
         "thstrm_amount": "70,000,000,000"},
    ], reprt="11011"), ticker="000270", statement_basis="CFS")

    assert _val(df, "CAPEX_PPE", "2025-12-31") == 50_000_000_000


# ── 재무상태표 ──────────────────────────────────────────────────────
def test_bs_uses_instant_fields():
    df = normalize_financials(_payload([{
        "sj_div": "BS", "account_id": "ifrs-full_EquityAttributableToOwnersOfParent",
        "thstrm_amount": "1,030,000,000,000", "frmtrm_amount": "1,000,000,000,000",
    }], reprt="11013", year="2026"), ticker="005930", statement_basis="CFS")

    eq = df[df.element == "EQUITY_CONTROLLING"]
    assert set(eq["period_type"]) == {"INSTANT"}
    assert _val(df, "EQUITY_CONTROLLING", "2026-03-31") == 1_030_000_000_000
    assert _val(df, "EQUITY_CONTROLLING", "2025-03-31") == 1_000_000_000_000
