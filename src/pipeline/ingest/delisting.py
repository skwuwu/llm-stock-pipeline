"""상장폐지 이력 — 생존편향의 유일한 해독제.

KIND 상장법인목록은 **현재 상장사만** 준다. 폐지된 회사는 목록에서 사라지고,
그 결과 security_master 에도 없다. 과거 시점으로 스크린을 돌리면 그때 상장돼
있던 회사 중 지금까지 살아남은 것만 보게 된다 — 백테스트 수익률이 구조적으로
부풀려진다.

FDR 의 `KRX-DELISTING` 이 폐지일과 사유를 준다(전체 4,174건, 주권 2,099건).

**합병 소멸을 전손으로 처리하면 반대 방향으로 틀린다.**
2026년 폐지 사례를 보면 성격이 완전히 다르다:

    더존비즈온   지주회사(최대주주등)의 완전자회사화 등   → 주주는 대가를 받았다
    신세계푸드   지주회사(최대주주등)의 완전자회사화 등   → 같음
    스타코링크   기업의 계속성 및 경영의 투명성 …        → 사실상 전손
    하나30호스팩 상장예비심사신청서 미제출 …            → 공모가 환급

셋을 같은 '폐지'로 묶으면 편향을 고치려다 새 편향을 만든다. 그래서 사유를
네 갈래로 분류한다.

`ToSymbol`(승계 종목)은 404건 중 12건만 채워져 있어 쓸 수 없다 — 사유 텍스트로 판정한다.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

# 우리가 다루는 것은 보통주다. 스팩·수익증권·신주인수권증서는 유니버스에
# 애초에 들어오지 않으므로(is_spac 가드) 폐지 이력도 필요 없다 —
# 다만 스팩은 '폐지'가 정상 수명 종료라 분류만 해두고 걸러낸다.
SECU_GROUP_STOCK = "주권"

# ── 사유 → 결과 분류 ────────────────────────────────────────────────
# 기준은 하나다: **주주에게 무슨 일이 일어났는가.**
# 순서가 규칙이다 — 앞엣것이 이긴다. 자진상폐를 '해산'보다 먼저 봐야
# "상장폐지 신청"이 failed 로 새지 않는다.
_MERGED = (
    "피흡수합병", "흡수합병", "합병으로", "완전자회사", "완전 자회사",
    "타법인의 완전자회사로 편입", "지주회사", "포괄적 주식교환", "주식의 포괄적",
)
# 자진상폐는 대개 공개매수를 동반한다 — 프리미엄을 받고 나간다.
# 합병과 묶지 않는 이유: 응하지 않은 주주는 비상장 주식을 떠안는다.
_VOLUNTARY = ("상장폐지 신청", "신청에 의한 상장폐지", "자진")
# 정상 수명 종료. 스팩은 공모가 환급, 선박투자회사 등은 존속기간 만료.
_DISSOLVED = ("스팩", "기업인수목적", "상장예비심사", "합병상장예비심사",
              "존속기간")
_FAILED = (
    "감사의견", "의견거절", "계속성", "경영의 투명성", "상장폐지기준",
    "해산", "파산", "부도", "자본전액잠식", "자본잠식",
    "매출액 미달", "시가총액 미달", "주식분산", "분산요건",
    "회생절차", "사업보고서", "감사범위 제한", "지정자문인",
)

MERGED, VOLUNTARY, DISSOLVED, FAILED, OTHER = (
    "merged", "voluntary", "dissolved", "failed", "other")

# 주주가 대가를 받은 결과. 백테스트에서 전손으로 세면 안 된다.
COMPENSATED = frozenset({MERGED, VOLUNTARY, DISSOLVED})


def classify_reason(reason: str, name: str = "") -> str:
    """폐지 사유 → 결과. 주주에게 무슨 일이 일어났는가로 가른다.

    merged      합병·완전자회사화. 대가를 받았으므로 전손이 아니다
    voluntary   자진상폐. 대개 공개매수 프리미엄. 다만 응하지 않은 주주는
                비상장 주식을 떠안아 합병과 성격이 다르다
    dissolved   스팩 해산·존속기간 만료. 정상 수명 종료, 환급
    failed      감사의견 거절·계속성 결여·해산. 사실상 전손
    other       분류 불가. **failed 로 밀어넣지 않는다** — 모르는 것을
                전손으로 세면 백테스트가 반대 방향으로 틀린다
    """
    # 사유가 비어 오는 행이 있다(NaN → float). str 로 강제하지 않으면 터진다.
    def _s(v) -> str:
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) \
            else " ".join(str(v).split())

    s, nm = _s(reason), _s(name)
    if not s:
        return DISSOLVED if "스팩" in nm else OTHER
    if any(k in s for k in _MERGED):
        return MERGED
    if any(k in s for k in _VOLUNTARY):
        return VOLUNTARY
    # 스팩은 사유가 '심사청구서 미제출'이라 실패처럼 보인다. 이름으로 가른다.
    if "스팩" in nm or any(k in s for k in _DISSOLVED):
        return DISSOLVED
    if any(k in s for k in _FAILED):
        return FAILED
    return OTHER


_COLS = ["ticker", "name", "market", "secu_group", "listing_date",
         "delisting_date", "reason", "outcome", "to_ticker", "to_name",
         "snapshot_date"]


def fetch_delistings(snapshot: date | None = None,
                     stock_only: bool = True) -> pd.DataFrame:
    """FDR KRX-DELISTING → 정규화된 폐지 이력.

    snapshot_date 를 남기는 이유: 이 목록도 시점 스냅샷이다. KRX 가 과거
    데이터를 정정할 수 있으므로 '언제 본 것인가'를 기록해 둔다.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import FinanceDataReader as fdr

    d = fdr.StockListing("KRX-DELISTING")
    if d is None or d.empty:
        return pd.DataFrame(columns=_COLS)

    d = d.rename(columns={
        "Symbol": "ticker", "Name": "name", "Market": "market",
        "SecuGroup": "secu_group", "ListingDate": "listing_date",
        "DelistingDate": "delisting_date", "Reason": "reason",
        "ToSymbol": "to_ticker", "ToName": "to_name"})
    if stock_only:
        d = d[d["secu_group"] == SECU_GROUP_STOCK]

    for c in ("listing_date", "delisting_date"):
        d[c] = pd.to_datetime(d[c], errors="coerce").dt.date
    # 폐지일이 없으면 PIT 판정을 할 수 없다. 0 이나 오늘로 메우지 않는다 —
    # 그러면 '아직 상장 중'과 '날짜 미상'이 같아진다.
    d = d[d["delisting_date"].notna()].copy()
    d["ticker"] = d["ticker"].astype(str).str.zfill(6)
    d["outcome"] = [classify_reason(r, n)
                    for r, n in zip(d["reason"], d["name"])]
    d["snapshot_date"] = snapshot or date.today()
    for c in ("to_ticker", "to_name"):
        d[c] = d[c].where(d[c].notna(), None)
    return d.reindex(columns=_COLS).drop_duplicates(
        subset=["ticker", "delisting_date"])


def coverage_report(delistings: pd.DataFrame, since: date | None = None) -> dict:
    """수집 결과 요약. 분류가 한쪽으로 쏠렸는지 보는 용도."""
    d = delistings
    if since is not None:
        d = d[d["delisting_date"] >= since]
    return {
        "rows": len(d),
        "by_outcome": d["outcome"].value_counts().to_dict(),
        "unclassified": int((d["outcome"] == OTHER).sum()),
        "earliest": str(d["delisting_date"].min()) if len(d) else None,
        "latest": str(d["delisting_date"].max()) if len(d) else None,
    }
