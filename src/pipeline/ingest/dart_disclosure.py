"""공시 목록 수집 — 촉매(catalyst)의 앵커.

`rcept_no` 없는 촉매는 존재할 수 없다. 이 모듈이 그 rcept_no 를 가져온다.

왜 목록만 받는가
────────────────
공시 **본문**은 건당 1콜인데, 19일치 거래소공시가 3,171건이다. 전부 받으면
연 6만 콜이고 대부분은 촉매가 아니다. 목록은 100건당 1콜이라 연 700콜이면
끝난다. 목록의 report_nm 으로 촉매 후보를 먼저 거른 뒤, **그것만** 본문을
받는 것이 순서다.

정정공시
────────
`[기재정정]주요사항보고서(자기주식취득결정)` 처럼 접두어가 붙어 온다.
접두어를 떼어 report_key 로 패턴 매칭하되, **정정이라는 사실은 버리지 않는다** —
촉매 검증의 첫 체크가 "이 건이 나중에 정정됐나" 이기 때문이다.
facts_financial 이 revision_of 로 원본을 보존하는 것과 같은 규율이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from pipeline.ingest.dart import DartClient

# 정정·연장 접두어. catalyst_v1.yaml 의 normalize.amendment_prefix 와 같은 것을
# 코드 쪽에도 두는 이유: 설정 파일이 없어도 수집은 돌아야 하기 때문이다.
AMENDMENT_PREFIX = re.compile(
    r"^\[(기재정정|첨부정정|첨부추가|연장결정|정정명령부과|변경등록|취소|기타정정)\]")

# 촉매가 사는 곳. A(정기공시)는 이미 다른 경로로 받고 있다.
CATALYST_TYPES = ("B", "I")

PAGE_COUNT = 100          # DART 최대치. 페이지당 1콜이므로 크게 잡는 게 이득이다
MAX_PAGES = 400           # 폭주 방지. 400페이지 = 4만 건이면 어떤 기간이든 충분하다


def normalize_report_name(report_nm: str) -> tuple[str, bool]:
    """(정정 접두어를 뗀 이름, 정정 여부)."""
    s = " ".join((report_nm or "").split())
    m = AMENDMENT_PREFIX.match(s)
    if not m:
        return s, False
    return AMENDMENT_PREFIX.sub("", s).strip(), True


@dataclass
class DisclosureIngestor:
    client: DartClient

    def fetch_range(self, bgn: date, end: date,
                    types: tuple[str, ...] = CATALYST_TYPES,
                    refresh: bool = False) -> pd.DataFrame:
        """기간 × 유형의 공시 목록 전체. 페이지를 끝까지 돈다.

        **부분 수집을 성공으로 오인하지 않는다.** total_page 를 다 돌지 못하면
        예외를 던진다 — 조용히 잘린 목록으로 촉매를 만들면 '그 종목은 공시가
        없었다'는 잘못된 결론이 나온다.
        """
        rows: list[dict] = []
        for ty in types:
            page, total_page = 1, None
            while page <= MAX_PAGES:
                r = self.client.disclosure_list(
                    bgn.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                    page=page, page_count=PAGE_COUNT, pblntf_ty=ty, refresh=refresh)
                st = r.get("status")
                if st == "013":            # 조회 결과 없음
                    break
                if st != "000":
                    raise RuntimeError(
                        f"공시 목록 실패 ty={ty} {bgn}~{end} p{page}: "
                        f"status={st} {r.get('message')}")
                total_page = int(r.get("total_page") or 1)
                for it in r.get("list", []):
                    key, amend = normalize_report_name(it.get("report_nm"))
                    rows.append({
                        "rcept_no": it["rcept_no"],
                        "ticker": (it.get("stock_code") or "").strip() or None,
                        "corp_code": it["corp_code"],
                        "corp_name": it.get("corp_name"),
                        "corp_cls": it.get("corp_cls"),
                        "report_nm": it.get("report_nm"),
                        "report_key": key,
                        "is_amendment": amend,
                        "pblntf_ty": ty,
                        "rcept_dt": pd.to_datetime(it["rcept_dt"],
                                                   format="%Y%m%d").date(),
                        "flr_nm": it.get("flr_nm"),
                    })
                if page >= total_page:
                    break
                page += 1
            else:
                raise RuntimeError(
                    f"공시 목록이 MAX_PAGES({MAX_PAGES})를 넘었다 ty={ty} "
                    f"{bgn}~{end} — 기간을 쪼갤 것. 잘린 목록으로 촉매를 만들면 "
                    f"'공시가 없었다'는 잘못된 결론이 나온다")
        if not rows:
            return pd.DataFrame(columns=[
                "rcept_no", "ticker", "corp_code", "corp_name", "corp_cls",
                "report_nm", "report_key", "is_amendment", "pblntf_ty",
                "rcept_dt", "flr_nm"])
        return pd.DataFrame(rows).drop_duplicates(subset=["rcept_no"])


def chunk_ranges(bgn: date, end: date, days: int = 14) -> list[tuple[date, date]]:
    """긴 기간을 잘라 페이지 폭주를 막는다.

    거래소공시는 하루 약 170건이라 14일이면 2,400건 = 24페이지다.
    한 번에 1년을 요청하면 600페이지가 되고 MAX_PAGES 에 걸린다.
    """
    out, cur = [], bgn
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        out.append((cur, stop))
        cur = stop + timedelta(days=1)
    return out
