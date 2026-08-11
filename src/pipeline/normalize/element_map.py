"""DART XBRL 표준계정코드 → 내부 정규 element 매핑.

미매핑을 조용히 0으로 만들지 않는다. 매핑되지 않은 계정은 그냥 버리고,
필수 element 가 빠지면 하류에서 품질 플래그로 드러나게 한다.
"""

from __future__ import annotations

# account_id → 내부 element. 앞에 오는 것이 우선(같은 element 에 여러 후보가 매칭되면
# priority 낮은 숫자가 이긴다).
ACCOUNT_MAP: dict[str, tuple[str, int]] = {
    # 손익
    "ifrs-full_Revenue":                                         ("REVENUE", 0),
    "ifrs-full_RevenueFromContractsWithCustomers":               ("REVENUE", 1),
    "ifrs-full_RevenueFromSaleOfGoods":                          ("REVENUE", 2),
    "dart_OperatingIncomeLoss":                                  ("OPERATING_INCOME", 0),
    "ifrs-full_ProfitLossFromOperatingActivities":               ("OPERATING_INCOME", 1),
    "ifrs-full_ProfitLoss":                                      ("NET_INCOME", 0),
    "ifrs-full_ProfitLossAttributableToOwnersOfParent":          ("NET_INCOME_CONTROLLING", 0),

    # 재무상태
    "ifrs-full_Assets":                                          ("ASSETS", 0),
    "ifrs-full_Equity":                                          ("EQUITY_TOTAL", 0),
    "ifrs-full_EquityAttributableToOwnersOfParent":              ("EQUITY_CONTROLLING", 0),
    "ifrs-full_CashAndCashEquivalents":                          ("CASH_AND_EQUIV", 0),
    # 유동비율용. 금융업은 유동/비유동 구분을 하지 않는 경우가 많아 결측이 흔하다 —
    # 결측을 0 으로 메우지 않고 그대로 두면 체크가 '계산 불가'로 넘어간다.
    "ifrs-full_CurrentAssets":                                   ("ASSETS_CURRENT", 0),
    "ifrs-full_CurrentLiabilities":                              ("LIABILITIES_CURRENT", 0),
    "dart_ShortTermBorrowings":                                  ("BORROWINGS_SHORT", 0),
    "ifrs-full_ShorttermBorrowings":                             ("BORROWINGS_SHORT", 1),
    "ifrs-full_LongtermBorrowings":                              ("BORROWINGS_LONG", 0),
    "dart_LongTermBorrowingsGross":                              ("BORROWINGS_LONG", 1),

    # 현금흐름
    "ifrs-full_CashFlowsFromUsedInOperatingActivities":          ("CFO", 0),

    # CAPEX 는 단일 계정이 아니라 여러 취득 항목의 합이다(AGGREGATE_ELEMENTS 참조).
    # 취득(유출)만 넣는다 — 처분(ProceedsFromSales*)이나 감가상각 조정
    # (AdjustmentsFor*)을 섞으면 FCF 가 조용히 부풀거나 깎인다.
    "ifrs-full_PurchaseOfPropertyPlantAndEquipment":             ("CAPEX_PPE", 0),
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities":
                                                                 ("CAPEX_PPE", 0),
    "dart_PurchaseOfOtherPropertyPlantAndEquipment":             ("CAPEX_PPE", 0),
    "dart_PurchaseOfStructure":                                  ("CAPEX_PPE", 0),
    "ifrs-full_PurchaseOfIntangibleAssetsOtherThanGoodwill":     ("CAPEX_INTANGIBLE", 0),
    "ifrs-full_PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities":
                                                                 ("CAPEX_INTANGIBLE", 0),
    "dart_PurchaseOfOtherIntangibleAssets":                      ("CAPEX_INTANGIBLE", 0),
}

# 이 element 들은 여러 계정의 '합'이다. 우선순위로 하나만 고르면 과소계상된다.
# (예: 유형자산의 취득 + 기타유형자산의 취득 + 시설장치의 취득을 각각 보고하는 회사)
AGGREGATE_ELEMENTS = {"CAPEX_PPE", "CAPEX_INTANGIBLE"}

# 지배주주 순이익이 없는 경우(별도재무제표 등) 총 순이익으로 대체 가능한 쌍.
# 대체가 일어나면 반드시 플래그를 남긴다 — 비지배지분이 큰 회사에서 PER 이 낮게 나온다.
CONTROLLING_FALLBACK = {
    "NET_INCOME_CONTROLLING": "NET_INCOME",
    "EQUITY_CONTROLLING": "EQUITY_TOTAL",
}

# 스크린이 성립하려면 반드시 있어야 하는 element.
REQUIRED_FOR_SCREEN = {"NET_INCOME_CONTROLLING", "EQUITY_CONTROLLING"}

# sj_div (재무제표 구분) → 기간 성격
#   BS  재무상태표   → 시점 잔액
#   IS  손익계산서 / CIS 포괄손익 / CF 현금흐름 → 기간 누적
SJ_PERIOD_KIND = {"BS": "INSTANT", "IS": "CUM", "CIS": "CUM", "CF": "CUM", "SCE": "INSTANT"}

# reprt_code → 누적 개월 수
REPRT_MONTHS = {"11013": 3, "11012": 6, "11014": 9, "11011": 12}
REPRT_QUARTER = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}


def map_account(account_id: str) -> tuple[str, int] | None:
    return ACCOUNT_MAP.get((account_id or "").strip())
