"""KIND(kind.krx.co.kr) 인제스천 — 상장사 마스터 + 관리종목.

왜 KRX 데이터포털이 아닌가:
  data.krx.co.kr 의 getJsonData.cmd 는 로그인을 요구하도록 바뀌었다(응답 "LOGOUT").
  pykrx 도 같은 엔드포인트를 쓰므로 전 함수가 막힌다 — 그런데 pykrx 는 예외를 던지지
  않고 **빈 리스트를 반환**한다. 그대로 배선했으면 "0종목"으로 조용히 통과했을 것이다.
  그래서 이 모듈은 결과가 비면 예외를 던진다.

KIND corpList 는 무인증으로 열려 있고 업종(KSIC 소분류)과 결산월을 함께 준다.
결산월은 회계기간 산정에 직접 필요하고, KRX 대분류보다 업종이 세밀하다.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

CORP_LIST_URL = "http://kind.krx.co.kr/corpgeneral/corpList.do"
ADMIN_ISSUE_URL = "http://kind.krx.co.kr/investwarn/adminissue.do"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MARKET_MAP = {"유가증권": "KOSPI", "코스닥": "KOSDAQ", "코넥스": "KONEX"}

# 관리종목 지정사유 중 감사의견 관련. 이 목록에 걸리면 audit_opinion_bad 로 본다.
AUDIT_OPINION_PATTERNS = re.compile(
    r"감사의견|감사범위\s*제한|의견\s*거절|부적정|한정")


class KindFetchError(RuntimeError):
    """소스가 비었거나 형태가 바뀌었다. 조용한 빈 결과를 허용하지 않는다."""


@dataclass
class KindClient:
    raw_root: Path
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _cached_get(self, name: str, url: str, params: dict | None,
                    snapshot: date, refresh: bool) -> bytes:
        """raw 는 스냅샷 일자별로 불변 보관. 이 소스는 '현재 상태'만 주므로
        스냅샷 일자가 곧 그 사실의 관측 시점이다."""
        p = self.raw_root / name / f"{snapshot.isoformat()}.html"
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not refresh:
            return p.read_bytes()
        r = self.session.get(url, params=params, headers=HEADERS, timeout=self.timeout_s)
        r.raise_for_status()
        if len(r.content) < 1000:
            raise KindFetchError(f"{name}: 응답이 너무 작다({len(r.content)}B). 소스 변경 의심.")
        p.write_bytes(r.content)
        return r.content

    def corp_list(self, snapshot: date | None = None, refresh: bool = False) -> pd.DataFrame:
        """전체 상장사. 반환 컬럼: ticker, name, market, industry, products,
        listing_date, fiscal_month, region"""
        snapshot = snapshot or date.today()
        content = self._cached_get("corpList", CORP_LIST_URL,
                                   {"method": "download", "searchType": "13"},
                                   snapshot, refresh)
        df = pd.read_html(io.BytesIO(content), encoding="euc-kr",
                          converters={"종목코드": str})[0]
        need = {"회사명", "종목코드", "업종", "상장일", "결산월"}
        if not need.issubset(df.columns):
            raise KindFetchError(f"corpList 컬럼 변경: {list(df.columns)}")
        if len(df) < 1000:
            raise KindFetchError(f"corpList 행 수 이상: {len(df)}")

        out = pd.DataFrame({
            "ticker": df["종목코드"].astype(str).str.strip().str.zfill(6),
            "name": df["회사명"].astype(str).str.strip(),
            "market": df.get("시장구분", pd.Series("", index=df.index))
                        .astype(str).str.strip().map(MARKET_MAP).fillna("UNKNOWN"),
            "industry": df["업종"].astype(str).str.strip(),
            "products": df.get("주요제품", pd.Series("", index=df.index)).astype(str),
            "listing_date": pd.to_datetime(df["상장일"], errors="coerce").dt.date,
            "fiscal_month": df["결산월"].astype(str).str.extract(r"(\d+)")[0]
                              .astype("Int64").fillna(12).astype(int),
            "region": df.get("지역", pd.Series("", index=df.index)).astype(str),
            "snapshot_date": snapshot,
        })
        return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    def admin_issues(self, snapshot: date | None = None,
                     refresh: bool = False) -> pd.DataFrame:
        """관리종목 현황. 반환: ticker, name, designation_date, reason,
        audit_opinion_bad, snapshot_date

        ⚠ PIT 한계: 이 소스는 **현재 지정 중인 종목의 스냅샷**이다.
        designation_date 가 있으므로 "언제부터 관리종목이었나"는 알 수 있지만,
        과거에 지정됐다가 **해제된 종목은 나타나지 않는다**.
        따라서 과거 as_of 질의에서 이 플래그는 과소 적용된다(누락 방향).
        정확한 과거 재현이 필요하면 스냅샷을 매일 적재해 이력을 직접 쌓아야 한다.
        """
        snapshot = snapshot or date.today()
        content = self._cached_get(
            "adminIssue", ADMIN_ISSUE_URL,
            {"method": "searchAdminIssueSub", "currentPageSize": "5000",
             "forward": "adminissue_down"},
            snapshot, refresh)
        df = pd.read_html(io.BytesIO(content), encoding="euc-kr",
                          converters={"종목코드": str})[0]
        if "종목코드" not in df.columns:
            raise KindFetchError(f"adminIssue 컬럼 변경: {list(df.columns)}")

        reason_col = next((c for c in df.columns if "사유" in c), None)
        date_col = next((c for c in df.columns if "지정일" in c or "일자" in c), None)
        reason = df[reason_col].astype(str) if reason_col else pd.Series("", index=df.index)

        return pd.DataFrame({
            "ticker": df["종목코드"].astype(str).str.strip().str.zfill(6),
            "name": df[[c for c in df.columns if "회사" in c or "종목명" in c][0]].astype(str),
            "designation_date": (pd.to_datetime(df[date_col], errors="coerce").dt.date
                                 if date_col else pd.NaT),
            "reason": reason,
            "audit_opinion_bad": reason.str.contains(AUDIT_OPINION_PATTERNS, na=False),
            "snapshot_date": snapshot,
        }).drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def parse_snapshot(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
