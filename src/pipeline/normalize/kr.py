"""DART 재무제표 응답 → PIT fact 행.

핵심 설계 두 가지.

1. **전기(frmtrm) 비교치도 fact 로 저장한다.**
   같은 응답 안에 들어 있고 reported_at 시점에 정당하게 알 수 있는 값이다.
   전년 동기 누적을 공짜로 얻으므로 TTM 계산에 추가 API 호출이 필요 없다
   (일일 20,000콜 한도에서 이건 큰 차이다).

2. **어느 필드가 '누적'인지는 재무제표 구분마다 다르다 — 실데이터로 확정했다.**
   아래 CUMULATIVE_IN_ADD_FIELD 주석에 근거와 검증 사례를 적어뒀다.
   손익계산서의 frmtrm_q_amount(3개월)를 현금흐름표의 frmtrm_q_amount(누적)와
   같게 취급하면 분기 PER 이 조용히 3~4배 어긋난다.
   amount_field 컬럼에 출처 필드를 남기므로, 규칙이 또 바뀌어도 재수집 없이
   `pipeline.cli renormalize` 로 고칠 수 있다.
"""

from __future__ import annotations

import calendar
import re
from datetime import date

import pandas as pd

from pipeline.ingest.dart import parse_rcept_dt
from pipeline.normalize.element_map import (
    AGGREGATE_ELEMENTS, REPRT_MONTHS, REPRT_QUARTER, SJ_PERIOD_KIND, map_account,
)
from pipeline.store.pit import FLOW_ELEMENTS, make_fact_id

DART_DOC_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={}"
_NUM = re.compile(r"^-?[\d,]+$")


def parse_amount(s) -> float | None:
    """DART 금액은 콤마 포함 문자열. '-' 나 빈칸은 결측(0 아님)."""
    if s is None:
        return None
    t = str(s).strip().replace(" ", "")
    if t in {"", "-", "0"} and t != "0":
        return None
    t = t.replace(",", "")
    if t.startswith("(") and t.endswith(")"):   # 괄호 음수 표기
        t = "-" + t[1:-1]
    try:
        return float(t)
    except ValueError:
        return None


def fiscal_end(bsns_year: int, months_cum: int, fiscal_month: int = 12) -> date:
    """사업연도 + 누적개월 → 회계기간 종료일.

    fiscal_month=12 (KRX 대부분)는 정확하다. 그 외 결산월은 근사이므로
    security_master.fiscal_month 를 채우고 non_standard_fiscal_month 플래그를 본다.
    """
    m = ((fiscal_month - 12 + months_cum - 1) % 12) + 1
    y = bsns_year if fiscal_month == 12 else bsns_year + (0 if m <= fiscal_month else -1)
    return date(y, m, calendar.monthrange(y, m)[1])


# 재무제표 구분마다 '누적'이 담기는 필드가 다르다. 실데이터로 확인한 규칙:
#
#   손익계산서(IS/CIS) 분기:
#     thstrm_amount     = 당분기 3개월      ← 누적 아님
#     thstrm_add_amount = 당기 누적          ← 이걸 써야 한다
#     frmtrm_q_amount   = 전년 동분기 3개월  ← 누적 아님
#     frmtrm_add_amount = 전년 동기 누적     ← 이걸 써야 한다
#     (삼성전자 3Q24 검증: thstrm 9.78조 / add 26.05조, 1Q 는 둘이 동일)
#
#   현금흐름표(CF) 분기: 애초에 누적으로만 보고된다.
#     thstrm_amount   = 당기 누적
#     frmtrm_q_amount = 전년 동기 '누적'  ← IS 와 의미가 정반대다
#     add 계열 필드는 존재하지 않는다
#     (기아 2025 검증: 1Q 3.01 → 반기 5.35 → 3Q 8.63 → FY 9.05조 단조증가)
#
#   재무상태표(BS): 시점 잔액. thstrm_amount / frmtrm_amount.
CUMULATIVE_IN_ADD_FIELD = {"IS", "CIS"}


def _pick_amount(row: dict, kind: str, is_annual: bool,
                 sj_div: str) -> tuple[float | None, str]:
    """당기 값. (값, 출처필드)"""
    if kind == "INSTANT" or is_annual:
        return parse_amount(row.get("thstrm_amount")), "thstrm_amount"
    if sj_div in CUMULATIVE_IN_ADD_FIELD:
        add = parse_amount(row.get("thstrm_add_amount"))
        if add is not None:
            return add, "thstrm_add_amount"
        # add 가 비면 thstrm 은 3개월이라 누적으로 쓸 수 없다. 만들지 않는다.
        return None, ""
    return parse_amount(row.get("thstrm_amount")), "thstrm_amount"


def _prior_amount(row: dict, kind: str, is_annual: bool,
                  sj_div: str) -> tuple[float | None, str]:
    """전기 비교치. 같은 응답에 들어 있고 reported_at 시점에 정당하게 알 수 있다."""
    if kind == "INSTANT" or is_annual:
        return parse_amount(row.get("frmtrm_amount")), "frmtrm_amount"
    if sj_div in CUMULATIVE_IN_ADD_FIELD:
        add = parse_amount(row.get("frmtrm_add_amount"))
        if add is not None:
            return add, "frmtrm_add_amount"
        # frmtrm_q_amount 는 3개월 값이다. 누적 자리에 넣으면 TTM 이 조용히 틀린다.
        return None, ""
    # CF: frmtrm_q_amount 가 전년 동기 누적
    q = parse_amount(row.get("frmtrm_q_amount"))
    if q is not None:
        return q, "frmtrm_q_amount"
    return parse_amount(row.get("frmtrm_amount")), "frmtrm_amount"


def normalize_financials(
    payload: dict,
    ticker: str,
    statement_basis: str,
    reported_at: date | None = None,
    available_lag_days: int = 1,
    fiscal_month: int = 12,
    market: str = "KR",
    revision_of: str | None = None,
) -> pd.DataFrame:
    """fnlttSinglAcntAll 응답 → facts_financial 행.

    available_lag_days 기본 1: 장중/장마감 후 공시 구분이 없으므로 익영업일부터
    사용 가능하다고 본다. 백테스트 보수성을 위한 선택이며 설정으로 조절한다.
    """
    rows = payload.get("list") or []
    if not rows:
        return pd.DataFrame()

    out: list[dict] = []
    seen: dict[tuple, int] = {}
    agg: dict[tuple, dict] = {}   # 합산 element 누적 버퍼   # (element, fiscal_end, period_type) -> priority

    for r in rows:
        mapped = map_account(r.get("account_id", ""))
        if mapped is None:
            continue                     # 미매핑은 버린다. 조용히 0으로 만들지 않는다.
        element, priority = mapped

        kind = SJ_PERIOD_KIND.get((r.get("sj_div") or "").strip())
        if kind is None:
            continue
        reprt = (r.get("reprt_code") or "").strip()
        months = REPRT_MONTHS.get(reprt)
        if months is None:
            continue
        is_annual = months == 12

        try:
            year = int(r["bsns_year"])
        except (KeyError, ValueError, TypeError):
            continue

        rcept_no = (r.get("rcept_no") or "").strip()
        rep_at = reported_at or (parse_rcept_dt(rcept_no[:8]) if len(rcept_no) >= 8 else None)
        if rep_at is None:
            continue
        avail_at = rep_at + pd.Timedelta(days=available_lag_days)
        avail_at = avail_at.date() if hasattr(avail_at, "date") else avail_at

        period_type = "INSTANT" if kind == "INSTANT" else ("FY" if is_annual else "CUM")
        quarter = REPRT_QUARTER.get(reprt)
        currency = (r.get("currency") or "KRW").strip()

        # 당기 + 전기를 각각 fact 로 적재
        sj_div = (r.get("sj_div") or "").strip()
        for amount_getter, y_offset, months_for_end in (
            (lambda: _pick_amount(r, kind, is_annual, sj_div), 0, months),
            (lambda: _prior_amount(r, kind, is_annual, sj_div), -1, months),
        ):
            value, field = amount_getter()
            if value is None:
                continue
            fy = year + y_offset
            fe = fiscal_end(fy, months_for_end, fiscal_month)
            key = (element, fe, period_type, statement_basis)

            if element in AGGREGATE_ELEMENTS:
                # 여러 취득 계정의 합. 부호 표기가 회사마다 달라 절대값으로 누적한다.
                if key in agg:
                    agg[key]["value"] += abs(value)
                    continue
                value = abs(value)
            else:
                # 같은 element 에 여러 계정이 매칭되면 priority 낮은 쪽이 이긴다
                if key in seen and seen[key] <= priority:
                    continue
                seen[key] = priority

            row = {
                "fact_id": make_fact_id(ticker, element, fe, period_type,
                                        statement_basis, rcept_no),
                "ticker": ticker,
                "market": market,
                "corp_code": (r.get("corp_code") or "").strip(),
                "fiscal_year": fy,
                "fiscal_quarter": quarter if period_type != "INSTANT" else quarter,
                "fiscal_end_date": fe,
                "period_type": period_type,
                "period_months": None if period_type == "INSTANT" else months_for_end,
                "reported_at": rep_at,
                "available_at": avail_at,
                "statement_basis": statement_basis,
                "element": element,
                "value": value,
                "currency": currency,
                "source_doc_id": rcept_no,
                "source_url": DART_DOC_URL.format(rcept_no),
                "revision_of": revision_of,
                "amount_field": field,
            }
            if element in AGGREGATE_ELEMENTS:
                agg[key] = row
            out.append(row)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    # 동일 키에 대해 priority 우선순위로 이미 걸렀지만, 방어적으로 한 번 더 dedup
    return df.drop_duplicates(subset=["fact_id"], keep="first").reset_index(drop=True)


# ── TTM 환산 ─────────────────────────────────────────────────────────
def compute_ttm(facts: pd.DataFrame, element: str) -> pd.DataFrame:
    """누적 보고 체계에서 TTM 을 복원한다.

    한국 재무제표는 분기 누적으로 보고된다. 3분기 보고서의 손익은 9개월 누적이므로
    그대로 쓰면 연환산이 틀린다.

        m == 12 :  TTM = FY(y)
        m <  12 :  TTM = FY(y-1) − CUM(y-1, m) + CUM(y, m)

    구성요소가 하나라도 없으면 값을 만들지 않고 reason 을 남긴다.
    빈칸을 0으로 메우면 PER 이 조용히 틀린다.
    """
    if element not in FLOW_ELEMENTS:
        raise ValueError(f"{element} 은 flow 계정이 아니다 — TTM 대상이 아님")

    f = facts[(facts["element"] == element) & (facts["period_type"].isin(["FY", "CUM"]))]
    if f.empty:
        return pd.DataFrame(columns=["ticker", "value", "ttm_basis", "ttm_reason"])

    results = []
    for ticker, g in f.groupby("ticker"):
        # 연결 우선. 기준이 섞이면 비교 불가라 하나로 고정한다.
        for basis in ("CFS", "OFS"):
            gb = g[g["statement_basis"] == basis]
            if not gb.empty:
                break
        else:
            continue

        gb = gb.sort_values(["fiscal_end_date", "reported_at"])
        latest = gb.iloc[-1]
        m = int(latest["period_months"])
        y = int(latest["fiscal_year"])

        if m == 12:
            results.append({"ticker": ticker, "value": float(latest["value"]),
                            "ttm_basis": basis, "ttm_reason": "annual"})
            continue

        prev_fy = gb[(gb["fiscal_year"] == y - 1) & (gb["period_months"] == 12)]
        prev_cum = gb[(gb["fiscal_year"] == y - 1) & (gb["period_months"] == m)]
        if prev_fy.empty or prev_cum.empty:
            missing = "prev_fy" if prev_fy.empty else "prev_cum"
            results.append({"ticker": ticker, "value": None,
                            "ttm_basis": basis, "ttm_reason": f"incomplete:{missing}"})
            continue

        ttm = (float(prev_fy.iloc[-1]["value"])
               - float(prev_cum.iloc[-1]["value"])
               + float(latest["value"]))
        results.append({"ticker": ticker, "value": ttm, "ttm_basis": basis,
                        "ttm_reason": f"rolled:{m}m"})

    return pd.DataFrame(results)


def compute_growth(facts: pd.DataFrame, element: str) -> pd.DataFrame:
    """전년 대비 성장률. **TTM 대 TTM 은 만들지 않는다.**

    직전연도 TTM 을 만들려면 2년 전 같은 분기 누적(예: 1Q24)이 필요한데,
    DART 정기공시 적재 범위에 그게 사실상 없다(실측 2026-08: 전체 3종목).
    없는 것을 있는 척하면 성장률이 소수 종목에만 붙고 나머지는 조용히
    탈락하므로, 실제로 계산 가능한 두 가지만 만든다.

        growth_fy : 최신 FY vs 그 전년 FY        — 안정적이나 최대 1년 묵는다
        growth_q  : 최신 누적분기 vs 전년 동기    — 신선하나 단일분기 노이즈

    **기저가 0 이하면 성장률을 만들지 않는다.** −10억 → +100억 은 −1100%
    라는 무의미한 수가 되고, 부호가 뒤집혀 있어 크기 비교도 성립하지 않는다.
    대신 그 경우를 `turnaround` 로 따로 표시한다 — 흑자전환은 성장 스크린이
    가장 놓치면 안 되는 유형인데, NaN 으로만 두면 조용히 빠진다.
    """
    if element not in FLOW_ELEMENTS:
        raise ValueError(f"{element} 은 flow 계정이 아니다 — 성장률 대상이 아님")

    # **증가율과 증가액을 같이 낸다.** 비율만 내면 기저가 작을 때 폭주한다 —
    # 전년 동기 영업이익이 1억이고 올해 30억이면 증가율 2,900% 다. 실측에서
    # 실적 서프라이즈 상위가 전부 이 유형이었다(해성디에스 2,995%).
    # 기저가 0 이하일 때만 막고 '아주 작은 양수'는 안 막았던 구멍이다.
    # 증가액을 시총으로 나누면 그 왜곡이 자동으로 사라진다.
    cols = ["ticker", "growth_fy", "growth_q", "delta_fy", "delta_q",
            "turnaround", "growth_reason"]
    f = facts[(facts["element"] == element) & (facts["period_type"].isin(["FY", "CUM"]))]
    if f.empty:
        return pd.DataFrame(columns=cols)

    def _rate(cur: float, base: float) -> float | None:
        return (cur - base) / base if base > 0 else None

    results = []
    for ticker, g in f.groupby("ticker"):
        # 기준을 섞으면 성장률이 회계기준 변경을 성장으로 읽는다. compute_ttm 과 동일.
        for basis in ("CFS", "OFS"):
            gb = g[g["statement_basis"] == basis]
            if not gb.empty:
                break
        else:
            continue
        gb = gb.sort_values(["fiscal_end_date", "reported_at"])

        row: dict = {"ticker": ticker, "growth_fy": None, "growth_q": None,
                     "delta_fy": None, "delta_q": None,
                     "turnaround": False, "growth_reason": basis}
        reasons = []

        # ── FY over FY ──────────────────────────────────────────────
        fy = gb[gb["period_months"] == 12]
        if fy.empty:
            reasons.append("no_fy")
        else:
            cur = fy.iloc[-1]
            y = int(cur["fiscal_year"])
            prev = fy[fy["fiscal_year"] == y - 1]
            if prev.empty:
                reasons.append("no_prev_fy")
            else:
                c, b = float(cur["value"]), float(prev.iloc[-1]["value"])
                row["growth_fy"] = _rate(c, b)
                # 증가액은 기저 부호와 무관하게 항상 의미가 있다 —
                # 적자 −10억 → 흑자 100억도 '110억 늘었다'로 셀 수 있다.
                row["delta_fy"] = c - b
                if b <= 0:
                    reasons.append("fy_base_nonpositive")
                    row["turnaround"] = bool(c > 0)

        # ── 최신 누적분기 over 전년 동기 ────────────────────────────
        cum = gb[(gb["period_type"] == "CUM") & gb["period_months"].notna()]
        if cum.empty:
            reasons.append("no_cum")
        else:
            cur = cum.iloc[-1]
            m, y = int(cur["period_months"]), int(cur["fiscal_year"])
            # 같은 누적 개월수끼리만 비교한다. 3개월 누적을 6개월 누적과 대면
            # 성장률이 아니라 기간 길이를 재게 된다.
            prev = cum[(cum["fiscal_year"] == y - 1) & (cum["period_months"] == m)]
            if prev.empty:
                reasons.append("no_prev_cum")
            else:
                c, b = float(cur["value"]), float(prev.iloc[-1]["value"])
                row["growth_q"] = _rate(c, b)
                row["delta_q"] = c - b
                if b <= 0:
                    reasons.append("q_base_nonpositive")
                    row["turnaround"] = row["turnaround"] or bool(c > 0)

        if reasons:
            row["growth_reason"] = f"{basis}:" + "|".join(reasons)
        results.append(row)

    return pd.DataFrame(results, columns=cols)


def latest_annual(facts: pd.DataFrame, element: str) -> pd.DataFrame:
    """연간(FY) 단일값의 최신치. 배당처럼 사업연도 단위로 확정되는 값에 쓴다.

    TTM 재구성도 잔액 조회도 맞지 않는다 — 배당은 누적도 잔액도 아니고
    '그 사업연도에 대해 한 번 확정되는 값' 이다.
    facts 는 이미 available_at <= as_of 로 걸러진 것이어야 한다(PIT 는 호출자 책임).
    """
    f = facts[(facts["element"] == element) & (facts["period_type"] == "FY")]
    if f.empty:
        return pd.DataFrame(columns=["ticker", "value", "fiscal_end_date"])
    f = f.sort_values(["ticker", "fiscal_end_date", "reported_at"],
                      ascending=[True, False, False])
    g = f.groupby("ticker", as_index=False).first()
    return g[["ticker", "value", "fiscal_end_date"]]


def latest_instant(facts: pd.DataFrame, element: str) -> pd.DataFrame:
    """잔액 계정의 최신 시점값. 연결 우선."""
    f = facts[(facts["element"] == element) & (facts["period_type"] == "INSTANT")]
    if f.empty:
        return pd.DataFrame(columns=["ticker", "value", "basis", "fiscal_end_date"])
    f = f.assign(_r=(f["statement_basis"] != "CFS").astype(int))
    f = f.sort_values(["ticker", "_r", "fiscal_end_date", "reported_at"],
                      ascending=[True, True, False, False])
    g = f.groupby("ticker", as_index=False).first()
    return g[["ticker", "value", "statement_basis", "fiscal_end_date"]].rename(
        columns={"statement_basis": "basis"})
