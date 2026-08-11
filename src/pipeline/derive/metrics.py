"""파생지표 + 품질 플래그. 오탐의 90%가 여기서 결정된다.

계산 규약(설계서 §3.2와 1:1 대응):
  PER  = 시총(보통주+우선주) / 지배주주 순이익 TTM
  PBR  = 시총(보통주+우선주) / 지배주주지분
  분모에 총자본/총순이익을 쓰면 비지배지분만큼 저평가로 보인다.
  분자에 보통주 시총만 쓰면 우선주 있는 회사가 저평가로 보인다. 둘 다 흔한 오류.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from pipeline.normalize.kr import (compute_growth, compute_ttm, latest_annual,
                                   latest_instant)

STALE_DAYS = 200
ONEOFF_RATIO = 1.5     # 순이익 > 영업이익 × 1.5 → 일회성 이익 의심
PER_OP_DIVERGENCE = 3.0

# 물리적 불가능 경계. '놀라운 값'이 아니라 '있을 수 없는 값'만 잡도록 넉넉히 둔다.
MAX_PLAUSIBLE_MARKET_CAP = 2_000e12   # 2,000조원 — KRX 최대 종목의 몇 배
MAX_PLAUSIBLE_PBR = 200.0
# 0.001 은 50배 느슨했다 — 시총이 자본의 2% 인 종목도 그냥 통과시켰다.
# 가드 통과 집합의 PBR p0.1% 가 0.051 이라 그 아래는 사실상 전부 데이터 오류다.
# 다만 자산주가 극단적으로 싸질 여지를 남겨 '불가능' 구간만 막는다.
MIN_PLAUSIBLE_PBR = 0.02

# ── 시총 정합성 (자본·매출 양쪽 대비) ────────────────────────────────
# PBR 만으로도 PSR 만으로도 가릴 수 없다:
#   · 저마진 유통·상사는 PSR 이 정상적으로 낮다(서원 0.028, KG케미칼 0.031)
#     — 대신 PBR 은 멀쩡하다(0.30, 0.29).
#   · 자산주는 PBR 이 낮을 수 있으나 PSR 은 정상이다.
# **둘 다 무너지면 시총 자체가 틀린 것이다.** 매출·자본은 DART 에서 오므로
# 주가와 완전히 독립이다 — 두 번째 가격 소스 없이 쓸 수 있는 유일한 교차검증이다.
# 실측: 포스코스틸리온 시총 277억 / 자본 3,841억 / 매출 1조 1,275억.
MAX_PBR_FOR_CAP_CHECK = 0.15
MAX_PSR_FOR_CAP_CHECK = 0.08

# ROIC 의 NOPAT 계산에 쓰는 실효세율. 종목별 실효세율을 쓰는 게 정확하지만
# 일회성 세무 항목 때문에 오히려 노이즈가 커져, 단일 가정을 쓰고 그 사실을 남긴다.
# 이건 스크린 임계값이 아니라 **모델링 가정**이라 설정이 아니라 상수로 둔다.
EFFECTIVE_TAX_RATE = 0.22

# ── 배당 역산 주가 교차검증 ──────────────────────────────────────────
# 주당배당금 ÷ 회사가 보고한 배당수익률 = 그 회사가 기준가로 삼은 주가.
# 우리 종가와 이게 크게 벌어지면 둘 중 하나가 틀렸다.
# 실측: 대한제분 DPS 4,000원 ÷ 3.0% = 153,846원인데 저장된 종가는 11,510원이었다.
# 배당 역산 '주식수'는 정확했으므로(배율 0.98) 틀린 쪽은 종가다.
# 종가가 과소하면 시총이 과소 → PER·PBR 이 인위적으로 싸 보여 스크린을 오염시킨다.
MIN_YIELD_FOR_PRICE_CHECK = 0.1    # 보고 수익률(%). 이보다 작으면 역산이 폭주한다
                                   # (영풍 DPS 5원/0.01% → 5만원 오차가 5억으로 증폭)
PRICE_DIVERGENCE_LIMIT = 5.0       # 실측 p99=3.8 / p01=0.28. 5배는 그 바깥이다
# 배당 데이터 자체가 물리적으로 불가능한 경우. 이땐 주가가 아니라 배당이 틀린 것이라
# 주가를 의심할 근거가 되지 않는다 — 정상 종목을 배제하지 않으려면 먼저 갈라내야 한다.
# 실측: 아비코전자 수익률 30%·DPS 30원(DART 가 같은 값을 두 칸에),
#       와이엔텍 DPS 18억원/주(배당총액을 주당란에 기입).
MAX_PLAUSIBLE_DIV_YIELD_PCT = 20.0


def build_metrics(facts: pd.DataFrame, prices: pd.DataFrame,
                  master: pd.DataFrame, as_of: date,
                  status: pd.DataFrame | None = None) -> pd.DataFrame:
    """as_of 시점 지표 테이블. facts 는 이미 PIT 필터링된 것이어야 한다.

    status: PitStore.status_asof() 결과. None 이면 상태 기반 가드가 전부 False 가
    되므로, 호출자는 그 사실을 알고 넘겨야 한다(CLI 는 항상 넘긴다).
    """
    if facts.empty or prices.empty:
        return pd.DataFrame()

    ni = compute_ttm(facts, "NET_INCOME_CONTROLLING").rename(
        columns={"value": "net_income_ttm", "ttm_reason": "ni_ttm_reason"})
    ni_total = compute_ttm(facts, "NET_INCOME").rename(
        columns={"value": "net_income_total_ttm"})[["ticker", "net_income_total_ttm"]]
    op = compute_ttm(facts, "OPERATING_INCOME").rename(
        columns={"value": "operating_income_ttm"})[["ticker", "operating_income_ttm"]]
    rev = compute_ttm(facts, "REVENUE").rename(
        columns={"value": "revenue_ttm"})[["ticker", "revenue_ttm"]]
    # 성장 스크린용. TTM 대 TTM 은 원천에 2년 전 분기가 없어 불가능하다(compute_growth 참조).
    rev_g = compute_growth(facts, "REVENUE").rename(columns={
        "growth_fy": "rev_growth_fy", "growth_q": "rev_growth_q",
        "delta_fy": "rev_delta_fy", "delta_q": "rev_delta_q",
        "turnaround": "rev_turnaround", "growth_reason": "rev_growth_reason"})
    op_g = compute_growth(facts, "OPERATING_INCOME").rename(columns={
        "growth_fy": "op_growth_fy", "growth_q": "op_growth_q",
        "delta_fy": "op_delta_fy", "delta_q": "op_delta_q",
        "turnaround": "op_turnaround", "growth_reason": "op_growth_reason"})
    cfo = compute_ttm(facts, "CFO").rename(
        columns={"value": "cfo_ttm", "ttm_reason": "cfo_ttm_reason"})[
        ["ticker", "cfo_ttm", "cfo_ttm_reason"]]
    capex_ppe = compute_ttm(facts, "CAPEX_PPE").rename(
        columns={"value": "capex_ppe_ttm"})[["ticker", "capex_ppe_ttm"]]
    capex_int = compute_ttm(facts, "CAPEX_INTANGIBLE").rename(
        columns={"value": "capex_intangible_ttm"})[["ticker", "capex_intangible_ttm"]]

    eq = latest_instant(facts, "EQUITY_CONTROLLING").rename(
        columns={"value": "equity_controlling", "basis": "equity_basis",
                 "fiscal_end_date": "equity_asof"})
    eq_total = latest_instant(facts, "EQUITY_TOTAL").rename(
        columns={"value": "equity_total"})[["ticker", "equity_total"]]
    assets = latest_instant(facts, "ASSETS").rename(
        columns={"value": "assets"})[["ticker", "assets"]]
    ca = latest_instant(facts, "ASSETS_CURRENT").rename(
        columns={"value": "assets_current"})[["ticker", "assets_current"]]
    cl = latest_instant(facts, "LIABILITIES_CURRENT").rename(
        columns={"value": "liabilities_current"})[["ticker", "liabilities_current"]]
    cash = latest_instant(facts, "CASH_AND_EQUIV").rename(
        columns={"value": "cash"})[["ticker", "cash"]]
    b_s = latest_instant(facts, "BORROWINGS_SHORT").rename(
        columns={"value": "borrow_short"})[["ticker", "borrow_short"]]
    b_l = latest_instant(facts, "BORROWINGS_LONG").rename(
        columns={"value": "borrow_long"})[["ticker", "borrow_long"]]
    # 배당은 연간 확정값이라 TTM 도 잔액도 아니다(latest_annual 참조).
    dps = latest_annual(facts, "DPS_CASH").rename(
        columns={"value": "dps_cash", "fiscal_end_date": "dps_fiscal_end"})
    payout = latest_annual(facts, "PAYOUT_RATIO_PCT").rename(
        columns={"value": "payout_ratio_pct"})[["ticker", "payout_ratio_pct"]]
    dy_rep = latest_annual(facts, "DIV_YIELD_REPORTED_PCT").rename(
        columns={"value": "div_yield_reported_pct"})[["ticker", "div_yield_reported_pct"]]
    dtot = latest_annual(facts, "DIVIDEND_TOTAL_MM").rename(
        columns={"value": "dividend_total_mm"})[["ticker", "dividend_total_mm"]]
    own = latest_annual(facts, "OWNER_STAKE_PCT").rename(
        columns={"value": "owner_stake_pct"})[["ticker", "owner_stake_pct"]]
    # LARGEST_HOLDER_PCT(개인 최대주주) fact 는 스토어에 남기되 파생으로 올리지 않는다.
    # 한국 지배구조에서는 가족·계열사 분산 보유가 흔해 개인 지분율만으로는
    # 경영권 안정성을 판정할 수 없다 — 판정은 owner_stake(특수관계인 합계)가 한다.

    df = prices[["ticker", "close", "market_cap_common", "market_cap_total",
                 "shares_common", "treasury_shares", "adtv_20d"]].copy()
    for part in (ni[["ticker", "net_income_ttm", "ni_ttm_reason"]], ni_total, op, rev,
                 rev_g, op_g,
                 cfo, capex_ppe, capex_int, eq, eq_total, assets, ca, cl,
                 cash, b_s, b_l, dps, payout, dy_rep, dtot, own):
        df = df.merge(part, on="ticker", how="left")

    # 지배주주 계정 폴백.
    # 종속기업이 없거나 별도재무제표만 있는 회사는 '지배기업 소유주지분'을 따로 보고하지
    # 않는다. 그런 경우 비지배지분이 없으므로 총액 = 지배주주분이다.
    # 다만 비지배지분이 실제로 큰데 분리 보고가 없는 경우엔 PER/PBR 이 낮게 나오므로
    # 폴백 사실을 플래그로 반드시 남긴다(minority_interest_large 와 함께 본다).
    df["ni_fallback_used"] = df["net_income_ttm"].isna() & df["net_income_total_ttm"].notna()
    df["net_income_ttm"] = df["net_income_ttm"].fillna(df["net_income_total_ttm"])
    df.loc[df["ni_fallback_used"], "ni_ttm_reason"] = "fallback:total"

    df["equity_fallback_used"] = df["equity_controlling"].isna() & df["equity_total"].notna()
    df["equity_controlling"] = df["equity_controlling"].fillna(df["equity_total"])

    # 시총 분자: 우선주 포함. market_cap_total 이 없으면 보통주로 폴백하고 플래그.
    mc = df["market_cap_total"].fillna(df["market_cap_common"])
    df["market_cap_used"] = mc

    # ── 밸류에이션 ────────────────────────────────────────────────
    df["per"] = _safe_div(mc, df["net_income_ttm"])
    df["pbr"] = _safe_div(mc, df["equity_controlling"])
    df["per_op"] = _safe_div(mc, df["operating_income_ttm"])
    # bps·eps_ttm 은 소비처가 생기면 되살린다. PBR·PER 이 같은 정보를 주가 대비로
    # 이미 담고 있어, 계산만 하고 아무도 안 읽는 상태였다(2026-08-07 정리).

    capex = (pd.to_numeric(df["capex_ppe_ttm"], errors="coerce").fillna(0).abs()
             + pd.to_numeric(df["capex_intangible_ttm"], errors="coerce").fillna(0).abs())
    capex_missing = df["capex_ppe_ttm"].isna() & df["capex_intangible_ttm"].isna()
    df["fcf"] = np.where(capex_missing | df["cfo_ttm"].isna(), np.nan, df["cfo_ttm"] - capex)
    df["fcf_yield"] = _safe_div(pd.Series(df["fcf"]), mc)
    _num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["net_cash"] = _num("cash") - _num("borrow_short") - _num("borrow_long")
    df["net_cash_to_cap"] = _safe_div(df["net_cash"], mc)
    df["roe"] = _safe_div(df["net_income_ttm"], df["equity_controlling"])

    # 영업이익률. 매출이 0 이하면(금융업 일부, 매출 미보고) 비율이 의미를 잃는다.
    # 0 으로 두면 '마진 0%' 로 읽혀 저마진 기업과 구분이 안 되므로 NaN 으로 둔다.
    _rev = pd.to_numeric(df["revenue_ttm"], errors="coerce")
    df["op_margin"] = np.where(
        _rev > 0, _safe_div(df["operating_income_ttm"], _rev), np.nan)
    # 성장률의 결측은 '성장이 없다'가 아니라 '기저가 적자라 계산 불가'인 경우가 많다.
    # 두 상태를 구분하지 않으면 흑자전환 기업이 저성장 기업과 함께 탈락한다.
    #
    # **영업이익 기준만 쓴다.** rev_turnaround 는 매출 기저가 0 이하일 때 켜지는데,
    # 매출이 음수인 회사는 없다 — 그건 흑자전환이 아니라 매출 미보고·0 매출,
    # 즉 데이터 결손이다. 같이 OR 하면 결손을 호재로 읽는다.
    df["turnaround"] = df["op_turnaround"] == True     # noqa: E712 — NaN/object 안전

    # ── 재무건전성·퀄리티 (체크 엔진이 참조하는 기반 컬럼) ────────────
    # 부채는 개별 계정을 다 매핑하는 대신 항등식으로 얻는다. 사채·리스부채·
    # 충당부채까지 일일이 매핑하면 누락분만큼 부채가 과소계상되는데,
    # 자산−자본은 정의상 빠짐이 없다.
    df["net_debt"] = -df["net_cash"]
    df["liabilities"] = _sub(df, "assets", "equity_total")
    df["debt_ratio"] = _safe_div(df["liabilities"], df["equity_total"])
    df["current_ratio"] = _safe_div(df["assets_current"], df["liabilities_current"])

    # ROIC = NOPAT / 투입자본. 순현금 기업은 투입자본이 0 이하로 갈 수 있어
    # 그 경우 값을 만들지 않는다(무한대로 튀면 상위 랭킹을 오염시킨다).
    invested = pd.to_numeric(df["equity_total"], errors="coerce") + df["net_debt"]
    nopat = pd.to_numeric(df["operating_income_ttm"], errors="coerce") * (1 - EFFECTIVE_TAX_RATE)
    df["roic"] = np.where(invested > 0, _safe_div(nopat, invested), np.nan)

    # 발생액 품질. 분모를 순이익으로 두면 순이익이 0 근처일 때 발산하므로
    # 총자산으로 정규화한다(Sloan 1996 형태). 값이 클수록 이익이 현금을 앞선다.
    df["accrual_ratio"] = _safe_div(
        pd.to_numeric(df["net_income_ttm"], errors="coerce")
        - pd.to_numeric(df["cfo_ttm"], errors="coerce"), df["assets"])
    # 직관적 형태도 함께 둔다 — 순이익이 양수일 때만 의미가 있다.
    ni_pos = pd.to_numeric(df["net_income_ttm"], errors="coerce")
    df["cfo_to_ni"] = np.where(ni_pos > 0, _safe_div(df["cfo_ttm"], ni_pos), np.nan)

    # 배당 역산 주가와의 괴리. 배당 데이터는 종가와 **독립적인 소스**라
    # 주가 오류를 잡는 유일한 교차검증 수단이다.
    _rep_y = pd.to_numeric(df["div_yield_reported_pct"], errors="coerce")
    _checkable = (pd.to_numeric(df["dps_cash"], errors="coerce") > 0) &                  (_rep_y >= MIN_YIELD_FOR_PRICE_CHECK)
    _implied = np.where(_checkable,
                        pd.to_numeric(df["dps_cash"], errors="coerce") / (_rep_y / 100.0),
                        np.nan)
    # 역산 주가를 컬럼으로 남긴다 — 괴리가 잡혔을 때 '얼마여야 했는가'를
    # 바로 볼 수 있어야 원인(주가 vs 배당)을 가릴 수 있다.
    df["implied_price_div"] = _implied
    df["price_divergence"] = _safe_div(df["close"], df["implied_price_div"])

    # 자사주 비중. shares_common 은 자기주식을 포함한 발행총수다(bps 계산 참조).
    df["treasury_ratio"] = _safe_div(df["treasury_shares"], df["shares_common"])

    # ── 주주환원 ──────────────────────────────────────────────────
    # 배당수익률은 회사가 보고한 값(자체 기준가)이 아니라 as_of 종가로 다시 계산한다.
    # 스크린 시점의 배당수익률이 필요한 것이지, 회사가 결산 무렵 기준가로 계산한
    # 값이 필요한 게 아니다. 보고값은 교차검증용으로 남긴다.
    df["div_yield"] = _safe_div(df["dps_cash"], df["close"])
    # _safe_div 는 분모를 Series 로 만들어 인덱스 정렬하므로 스칼라를 넘기면 안 된다
    # (1개짜리 Series 가 되어 첫 행만 맞고 나머지가 NaN 이 된다).
    df["payout_ratio"] = pd.to_numeric(df["payout_ratio_pct"], errors="coerce") / 100.0
    # 보고값과의 괴리는 **원인을 나눠야** 쓸모가 있다. 하나로 뭉치면
    # '주가가 움직였다'(정상)와 '수치가 안 맞는다'(오류)가 구별되지 않는다.
    #
    # (1) 주가 하락으로 부풀려진 수익률 — 배당 트랩 후보.
    #     보고 수익률은 결산 무렵 기준가 기준이므로, 지금 계산값이 그보다 훨씬 높다는 건
    #     그 사이 주가가 빠졌다는 뜻이다. 과거 배당이 미래 배당을 보장하지 않으므로
    #     실적이 꺾인 저PER 고배당은 배당 컷 위험이 크다.
    _dy_rep = pd.to_numeric(df["div_yield_reported_pct"], errors="coerce") / 100.0
    # 임계값은 설정에서 정한다 — 여기서는 수치만 만든다.
    # div_yield_gap 은 체크가 expr(div_yield - div_yield_reported)로 직접 계산하도록
    # 바뀌면서 소비처를 잃었다. 두 곳에서 같은 식을 유지하면 한쪽만 고쳐진다.
    df["div_yield_reported"] = _dy_rep

    # (2) DART 자체 정합성 — 주당배당금 × 배당대상주식수 vs 보고된 배당총액.
    #     실측: 화천기공은 배당성향(25.13%)·순이익·DPS 가 서로 맞는데 총액만
    #     1.51배로 어긋난다. 유배당 1,201종목 중 23건(1.9%)이 이런 상태다.
    _base = (pd.to_numeric(df["shares_common"], errors="coerce")
             - pd.to_numeric(df["treasury_shares"], errors="coerce").fillna(0))
    df["dividend_total_implied"] = pd.to_numeric(df["dps_cash"], errors="coerce") * _base
    _tot_rep = pd.to_numeric(df["dividend_total_mm"], errors="coerce") * 1e6
    df["dividend_total_gap"] = (
        (df["dividend_total_implied"] - _tot_rep).abs() / _tot_rep.replace(0, np.nan))

    # (3) 지속가능성 — 과거 배당이 미래 배당을 보장하지 않는다.
    #     FCF 로 못 덮는 배당은 현금·차입으로 메우는 것이라 유지되기 어렵다.
    #     총액은 위에서 본 대로 신뢰도가 낮아, **DPS 로 역산한 값**을 쓴다.
    df["dividend_fcf_cover"] = _safe_div(df["fcf"], df["dividend_total_implied"])

    # ── 지배구조 ──────────────────────────────────────────────────
    # 지배구조 안정성의 지표는 개인 최대주주가 아니라 **특수관계인 합계**다.
    # 개인 지분만 보면 가족·계열사로 분산 보유한 회사가 취약해 보인다
    # (실측: 화천기공 본인 17.93% / 합계 48.78%).
    df["owner_stake"] = pd.to_numeric(df["owner_stake_pct"], errors="coerce") / 100.0

    master_cols = ["ticker", "name", "sector_code", "is_financial", "is_holding",
                   "fiscal_month", "is_spac", "is_reit", "is_preferred"]
    df = df.merge(master.reindex(columns=master_cols), on="ticker", how="left")
    for c in ("is_spac", "is_reit", "is_preferred"):
        df[c] = df[c].fillna(False).astype(bool)
    df["as_of"] = as_of
    return add_quality_flags(df, facts, as_of, status)


# 상태 이벤트 → 플래그 컬럼. 소스가 없으면 컬럼은 생기되 전부 False 다.
STATUS_FLAGS = ["admin_issue", "audit_opinion_bad_admin"]


def add_quality_flags(df: pd.DataFrame, facts: pd.DataFrame, as_of: date,
                      status: pd.DataFrame | None = None) -> pd.DataFrame:
    """하드 배제 플래그와 소프트 플래그. 스크린 엔진이 이 컬럼들만 본다."""
    # ── 하드 ──
    # 자본잠식: equity <= 0 이면 PBR 이 음수가 되어 'PBR < 1' 을 그냥 통과한다.
    df["capital_impairment"] = df["equity_controlling"].le(0).fillna(False)
    df["negative_earnings"] = df["net_income_ttm"].le(0).fillna(False)

    # 원본 데이터가 물리적으로 불가능한 값을 줄 때가 있다.
    # 실례: LS에코에너지의 DART 주식총수가 30.6조 주로 보고돼 시총이 1.4해원이 됐다.
    # 하한(min_market_cap)만 두면 이런 '너무 큰' 오류는 그대로 통과한다.
    df["market_cap_implausible"] = df["market_cap_used"].gt(MAX_PLAUSIBLE_MARKET_CAP).fillna(False)
    # PBR 은 외부 소스 없이 시총과 자본의 정합성을 교차검증한다.
    # 놀라운 값이 아니라 불가능한 값만 잡는다(정상 고평가주를 죽이지 않도록 넉넉히).
    df["pbr_implausible"] = (df["pbr"].gt(MAX_PLAUSIBLE_PBR)
                             | df["pbr"].between(0, MIN_PLAUSIBLE_PBR, inclusive="neither")
                             ).fillna(False)
    # 배당 데이터가 물리적으로 불가능한 경우를 **먼저** 갈라낸다.
    # 이건 주가가 아니라 배당이 틀린 것이라, 주가를 의심할 근거가 되지 못한다.
    df["dividend_data_impossible"] = (
        _col(df, "div_yield_reported_pct").gt(MAX_PLAUSIBLE_DIV_YIELD_PCT)
        | _col(df, "dps_cash").gt(_col(df, "close"))).fillna(False)

    # 남은 괴리는 **어느 쪽이 틀렸는지 알 수 없는** 상태다.
    # 배당은 종가와 독립된 소스라, 시총·PBR 만으로는 안 잡히는 주가 오류를
    # 여기서만 잡을 수 있다(실측: 대한제분은 배당 역산 주식수로 주가 오류가 확정됐고,
    # 포스코스틸리온은 PBR 0.072 로 스크린을 통과할 뻔했다).
    # 진짜 급락도 함께 걸린다. 그래도 배제하는 이유는 **이 지점에서 둘을 구별할
    # 수단이 없기 때문**이다 — 모르는 것을 통과시키지 않는다(시총 결측 배제와 동일).
    # 시총이 자본·매출 **양쪽** 대비 극단적으로 작으면 시총(=주가×주식수)이 틀렸다.
    # 금융업은 '매출' 개념이 달라 제외한다.
    # PSR 을 컬럼으로 남긴다 — 가드가 걸렸을 때 '자본 대비인가 매출 대비인가'를
    # 바로 봐야 원인을 가릴 수 있다.
    df["psr"] = _safe_div(df["market_cap_used"], _col(df, "revenue_ttm"))
    _fin = df["is_financial"].fillna(False).astype(bool) if "is_financial" in df.columns         else pd.Series(False, index=df.index)
    df["market_cap_inconsistent"] = (
        _col(df, "pbr").gt(0) & _col(df, "pbr").lt(MAX_PBR_FOR_CAP_CHECK)
        & df["psr"].gt(0) & df["psr"].lt(MAX_PSR_FOR_CAP_CHECK) & ~_fin).fillna(False)

    _pd = _col(df, "price_divergence")
    df["price_dividend_inconsistent"] = (
        (_pd.gt(PRICE_DIVERGENCE_LIMIT) | _pd.lt(1 / PRICE_DIVERGENCE_LIMIT))
        & ~df["dividend_data_impossible"]).fillna(False)

    last_report = (facts.groupby("ticker")["reported_at"].max()
                   .rename("last_reported_at").reset_index())
    df = df.merge(last_report, on="ticker", how="left")
    df["last_reported_at"] = pd.to_datetime(df["last_reported_at"]).dt.date
    df["stale_financials"] = df["last_reported_at"].apply(
        lambda d: True if pd.isna(d) else (as_of - d) > timedelta(days=STALE_DAYS))

    # 두 결함을 분리한다. 원인과 대응이 다르다:
    #   financials_missing = 재무 fact 자체가 없음      → 인제스천 커버리지 문제
    #   ttm_incomplete     = fact 는 있으나 구성요소 결측 → 이력 부족(신규 상장 등)
    for c in ("ni_fallback_used", "equity_fallback_used"):
        if c not in df:
            df[c] = False
    reason = df["ni_ttm_reason"]
    df["financials_missing"] = reason.isna()
    df["ttm_incomplete"] = reason.fillna("").str.startswith("incomplete")

    # 상태 기반 가드. status 를 안 넘기면 컬럼은 생기되 전부 False —
    # 스크린이 컬럼 존재만으로 '가드가 돈다'고 착각하지 않도록
    # status_source_missing 을 함께 남긴다.
    for flag in STATUS_FLAGS:
        df[flag] = False
    if status is not None and not status.empty:
        for flag in STATUS_FLAGS:
            hit = set(status.loc[status["status"] == flag, "ticker"])
            df[flag] = df["ticker"].isin(hit)
    df["status_source_missing"] = status is None or status.empty

    # ── 소프트 (통과시키되 표기하고 LLM 에도 전달) ──
    df["oneoff_profit_suspect"] = (
        df["net_income_ttm"] > df["operating_income_ttm"] * ONEOFF_RATIO).fillna(False)
    df["per_op_divergence"] = (
        _safe_div(df["per_op"], df["per"]) > PER_OP_DIVERGENCE).fillna(False)
    # 비지배지분이 큰데 지배주주 기준을 못 쓴 경우 PER/PBR 이 낮게 나온다
    df["minority_interest_large"] = (
        _safe_div(df["equity_total"] - df["equity_controlling"],
                  df["equity_total"]) > 0.20).fillna(False)
    df["cfs_missing_used_ofs"] = df.get("equity_basis", pd.Series("CFS", index=df.index)).eq("OFS")
    df["capex_unmapped"] = df["fcf"].isna() & df["cfo_ttm"].notna()
    df["non_standard_fiscal_month"] = df["fiscal_month"].fillna(12).ne(12)
    # 섹터 미매핑이면 V2 섹터 정합성 검증이 그 종목에 대해 아무 일도 하지 않는다.
    # '검증이 돌고 있다'는 착각을 막기 위해 명시적으로 남긴다.
    df["unmapped_industry"] = df["sector_code"].isna()
    df["financial_sector"] = df["is_financial"].fillna(False)
    df["holding_company"] = df["is_holding"].fillna(False)

    # 금융업은 FCF·순현금 개념이 성립하지 않는다 — 계산하지 않고 비운다
    fin = df["financial_sector"].astype(bool)
    df.loc[fin, ["fcf", "fcf_yield", "net_cash", "net_cash_to_cap"]] = np.nan
    return df


def _safe_div(a, b):
    """a / b. 0 나눗셈과 무한대는 NaN.

    스칼라 분모를 받으면 반드시 브로드캐스트해야 한다. pd.Series(100.0) 은
    **1행짜리 Series** 가 되어 인덱스 정렬로 첫 행만 계산되고 나머지가 전부 NaN 이
    된다(실측: payout_ratio 가 통째로 NaN 이었다). 이 함수는 파생지표 전체가
    통과하는 길목이라 조용한 오작동의 파급이 크다.
    """
    a = pd.Series(a).astype(float)
    if np.isscalar(b):
        b = pd.Series(float(b), index=a.index)
    else:
        b = pd.Series(b).astype(float)
        if not a.index.equals(b.index):
            b = b.reindex(a.index)
    return (a / b.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """컬럼이 없으면 전부 NaN 인 시리즈. **선택적 소스에만 쓴다.**

    add_quality_flags 는 배당 같은 선택적 데이터 없이도 호출될 수 있다.
    없는 것을 위반으로 단정하면 데이터 미수집이 곧 배제가 되므로, 없으면
    '판정 불가' → 플래그 False 로 흘린다(_hit 의 결측 처리와 같은 규율).
    필수 컬럼에는 쓰지 말 것 — 그건 MissingGuardError 로 실패해야 한다.
    """
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[name], errors="coerce")


def _sub(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """a − b. 어느 쪽이든 결측이면 결과도 결측 — 0 으로 메우지 않는다.

    자산이나 자본이 없는데 0 으로 채우면 부채비율이 그럴듯한 숫자로 나와
    가드를 통과해 버린다. 모르는 것은 모르는 채로 둔다.
    """
    return (pd.to_numeric(df[a], errors="coerce")
            - pd.to_numeric(df[b], errors="coerce"))


# 이 컬럼들이 통째로 비면 파생 단계가 고장난 것이다. 조용히 넘어가면
# 스크린은 '해당 없음' 으로 전 종목을 통과시키거나 전 종목을 탈락시킨다.
CRITICAL_METRICS = [
    "per", "pbr", "per_op", "fcf", "fcf_yield",
    "net_cash", "roe", "roic", "debt_ratio", "liabilities",
    "accrual_ratio", "treasury_ratio", "market_cap_used",
    # 성장·마진. GARP/퀄리티 스크린의 게이트 지표라 통째로 비면 그 스크린이
    # 조용히 '전 종목 탈락' 이 된다 — 결과가 0건이면 원인을 찾기 어렵다.
    "op_margin", "rev_growth_fy", "op_growth_fy", "rev_growth_q", "op_growth_q",
    # 증가**액**. 비율은 기저가 작으면 폭주해 랭킹을 오염시킨다 — 촉매 C3 가
    # 이걸 시총으로 나눠 쓴다(configs/catalysts/catalyst_v1.yaml).
    "op_delta_q", "op_delta_fy", "rev_delta_q", "rev_delta_fy",
]


class MetricsIntegrityError(RuntimeError):
    """파생지표가 통째로 비었다. 스크린을 돌리기 전에 실패시킨다."""


def assert_metrics_sane(df: pd.DataFrame,
                        extra: list[str] | None = None) -> dict[str, float]:
    """전부-NaN 컬럼을 잡고, 컬럼별 결측률을 돌려준다.

    _safe_div 의 스칼라 분모 버그처럼 '계산은 도는데 결과가 전부 NaN' 인 고장은
    예외를 던지지 않아 테스트가 늘어도 잡히지 않는다. 산출물 자체를 검사해야 한다.
    """
    if df.empty:
        raise MetricsIntegrityError("파생지표 테이블이 비었다")
    rates: dict[str, float] = {}
    dead: list[str] = []
    for c in CRITICAL_METRICS + list(extra or []):
        if c not in df.columns:
            dead.append(f"{c}(컬럼 없음)")
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        rates[c] = round(float(v.isna().mean()), 4)
        if v.isna().all():
            dead.append(f"{c}(전부 NaN)")
    if dead:
        raise MetricsIntegrityError(
            "파생지표가 통째로 비었다: " + ", ".join(dead)
            + ". 계산식이나 입력 컬럼을 확인할 것.")
    return rates


def data_defect_rate(df: pd.DataFrame) -> dict:
    """릴리스마다 기록할 KPI. 이 파이프라인의 '정상 동작' 정의."""
    n = len(df)
    if n == 0:
        return {"n": 0}
    hard = ["capital_impairment", "negative_earnings", "stale_financials",
            "ttm_incomplete", "financials_missing",
            "admin_issue", "audit_opinion_bad_admin",
            "market_cap_implausible", "pbr_implausible"]
    soft = ["oneoff_profit_suspect", "per_op_divergence", "minority_interest_large",
            "cfs_missing_used_ofs", "capex_unmapped", "non_standard_fiscal_month",
            "unmapped_industry", "status_source_missing",
            "ni_fallback_used", "equity_fallback_used"]
    return {
        "n": n,
        "hard": {c: int(df[c].sum()) for c in hard if c in df},
        "soft": {c: int(df[c].sum()) for c in soft if c in df},
        "per_computable": int(df["per"].notna().sum()),
        "pbr_computable": int(df["pbr"].notna().sum()),
    }
