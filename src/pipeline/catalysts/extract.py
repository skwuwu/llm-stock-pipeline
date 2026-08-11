"""촉매의 크기(magnitude)를 **결정론적으로** 뽑는다.

LLM 을 쓰지 않는다. 세 경로 전부 숫자가 원천에 이미 있다:

  구조화 API   C1 자사주, X1 희석
               DART 주요사항보고서 API 가 JSON 으로 금액을 준다.
               tsstkAqDecsn.aqpln_prc_ostk, cvbdIsDecsn.bd_fta 등.
  문서 정규식   C2 수주
               거래소공시는 서식이 고정이라 '계약금액 총액(원)' 라벨이 항상 있다.
               DART 가 '매출액 대비(%)' 까지 계산해 준다 — 우리가 나눌 필요도 없다.
  자체 계산     C3 실적
               잠정실적 공시를 파싱하지 않고 **우리 op_growth_q 를 쓴다.**
               공시는 '언제 알려졌나'(날짜 앵커)로만 쓴다. 이유는 두 가지다:
                 · 파싱 대상이 3,270건이라 문서 수집 비용이 크다
                 · 같은 수치를 두 경로로 만들면 어긋날 때 어느 쪽이 맞는지
                   판정할 근거가 없다. 하나로 고정하는 편이 정직하다

**magnitude 를 못 구하면 촉매를 만들지 않는다.** 0 으로 채우면 '작은 촉매'와
'크기를 모르는 촉매'가 같아진다 — 이 파이프라인이 반복해서 지켜온 규칙이다
(세그먼트 미공시를 0% 로 읽지 않는 것과 같다).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from pipeline.ingest.dart import DartClient

# ── 구조화 API: (엔드포인트, 금액필드들) ─────────────────────────────
# 금액필드는 **순서대로** 시도한다. 앞엣것이 '-' 면 다음 것.
# 실측: 자기주식취득결정에서 aqpln_prc_ostk 가 '-' 이고 주식수만 있는 경우가 있다.
STRUCTURED: dict[str, list[tuple[str, list[str], str | None]]] = {
    # (endpoint, 금액 후보 필드, 주식수 후보 필드)
    "C1": [
        ("tsstkAqDecsn", ["aqpln_prc_ostk"], "aqpln_stk_ostk"),
        ("tsstkAqTrctrCnsDecsn", ["ctr_prc"], None),
    ],
    "X1": [
        # 유상증자는 자금조달 목적별로 쪼개져 있어 **합산**해야 총액이 된다.
        ("piicDecsn", ["fdpp_fclt", "fdpp_bsninh", "fdpp_op",
                       "fdpp_dtrp", "fdpp_ocsa", "fdpp_etc"], "nstk_ostk_cnt"),
        ("cvbdIsDecsn", ["bd_fta"], None),
    ],
    "X2": [("tsstkDpDecsn", ["dppln_prc_ostk"], "dppln_stk_ostk")],
}
# 합산해야 하는 엔드포인트. 나머지는 '첫 유효값'.
SUM_FIELDS = {"piicDecsn"}

_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def _num(v) -> float | None:
    """'53,500,000,000' → 5.35e10. '-' 나 빈칸은 None(0 아님)."""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s or s in {"-", "―", "–"}:
        return None
    try:
        return float(s)
    except ValueError:
        m = _NUM.search(s)
        return float(m.group().replace(",", "")) if m else None


@dataclass
class Magnitude:
    value: float | None                 # 정규화 전 절대금액(원) 또는 비율
    basis: str                          # 어디서 왔나. 감사에 필요하다
    expires_at: date | None = None      # 공시가 종료일을 주면 그것을 쓴다
    fields: dict = field(default_factory=dict)


# ── 구조화 API 경로 ──────────────────────────────────────────────────
def from_structured(client: DartClient, kind: str, corp_code: str,
                    bgn: date, end: date, close: float | None = None,
                    ) -> dict[str, Magnitude]:
    """rcept_no → Magnitude. 기간 내 해당 종목의 모든 건을 한 번에 받는다."""
    out: dict[str, Magnitude] = {}
    for ep, amount_fields, share_field in STRUCTURED.get(kind, []):
        try:
            r = client._get_json(
                ep, {"corp_code": corp_code, "bgn_de": bgn.strftime("%Y%m%d"),
                     "end_de": end.strftime("%Y%m%d")},
                cache_key=f"{ep}_{corp_code}_{bgn:%Y%m%d}_{end:%Y%m%d}")
        except Exception:                                    # noqa: BLE001
            continue                                          # 없는 엔드포인트·권한
        if r.get("status") != "000":
            continue
        for it in r.get("list") or []:
            rc = it.get("rcept_no")
            if not rc:
                continue
            vals = [_num(it.get(f)) for f in amount_fields]
            vals = [v for v in vals if v is not None]
            if ep in SUM_FIELDS:
                amt = sum(vals) if vals else None
                basis = f"{ep}:sum({'+'.join(amount_fields)})"
            else:
                amt = vals[0] if vals else None
                basis = f"{ep}:{amount_fields[0]}"
            if amt is None and share_field and close:
                # 금액을 안 밝히고 주식수만 준 경우. **추정치라고 표시한다** —
                # 공시된 금액과 우리가 곱한 값은 같은 신뢰도가 아니다.
                n = _num(it.get(share_field))
                if n:
                    amt, basis = n * close, f"{ep}:{share_field}×close(추정)"
            if amt is None:
                continue
            out[rc] = Magnitude(value=amt, basis=basis,
                                fields={k: it.get(k) for k in
                                        (*amount_fields, share_field) if k})
    return out


# ── 문서 정규식 경로 (C2 수주) ───────────────────────────────────────
# 거래소공시는 서식이 고정이라 라벨이 항상 같은 자리에 있다.
# DART 가 '매출액 대비(%)' 를 이미 계산해 주므로 우리가 나누지 않는다 —
# 분모(최근 매출액)를 우리 revenue_ttm 으로 바꾸면 별도/연결이 어긋난다.
# to_plain_text 가 표 셀을 ' | ' 로 잇는다. 라벨과 값 사이에 항상 구분자가
# 끼므로 정규식이 그걸 건너뛸 수 있어야 한다 — 안 넣으면 전부 None 이 된다.
_SEP = r"[\s|]*"
_C2_TOTAL = re.compile(r"계약금액\s*총액\s*\(?원\)?" + _SEP + r"([\d,]+)")
_C2_VS_REV = re.compile(r"매출액\s*대비\s*\(?%\)?" + _SEP + r"([\d,]+\.?\d*)")
_C2_END = re.compile(r"종료일" + _SEP + r"(\d{4})[-.\s]*(\d{1,2})[-.\s]*(\d{1,2})")


def from_contract_document(text: str) -> Magnitude | None:
    """단일판매·공급계약 공시 본문 → 매출액 대비 비율.

    text 는 to_plain_text 결과여야 한다. <style> 이 남아 있으면 CSS 안의
    숫자가 잡힐 수 있다 — dart_document._STYLE_BLOCK 참조.
    """
    t = " ".join((text or "").split())
    m = _C2_VS_REV.search(t)
    if not m:
        return None
    pct = _num(m.group(1))
    if pct is None:
        return None
    exp = None
    if e := _C2_END.search(t):
        try:
            exp = date(int(e.group(1)), int(e.group(2)), int(e.group(3)))
        except ValueError:
            exp = None
    total = _num(m2.group(1)) if (m2 := _C2_TOTAL.search(t)) else None
    return Magnitude(value=pct / 100.0, basis="document:매출액대비(%)",
                     expires_at=exp, fields={"계약금액총액": total})


# ── 시효 ─────────────────────────────────────────────────────────────
def resolve_expiry(occurred_at: date, expires_days: int | None,
                   from_document: date | None = None) -> date | None:
    """공시가 종료일을 주면 그것이 우선. 없으면 카탈로그의 기본 일수.

    LLM 이 정하지 않는다 — 시효는 규칙이지 판단이 아니다.
    """
    if from_document is not None:
        return from_document
    if expires_days is None:
        return None
    return occurred_at + timedelta(days=expires_days)


def normalize(value: float | None, denominator: float | None) -> float | None:
    """절대금액 → 비율. 분모가 없거나 0 이하면 만들지 않는다."""
    if value is None or denominator is None:
        return None
    d = float(denominator)
    if not d or d <= 0 or pd.isna(d):
        return None
    return value / d
