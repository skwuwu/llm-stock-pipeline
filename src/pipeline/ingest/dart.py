"""DART OpenAPI 인제스천.

원칙:
  - raw 는 불변. API 응답 원문을 그대로 저장하고, 정규화는 항상 raw 에서 재생성한다.
    파싱 버그를 나중에 고칠 수 있는지가 여기서 결정된다.
  - 일일 20,000콜 한도. 캐시 히트는 호출로 세지 않으며, 남은 호출 수를 디스크에 기록한다.
  - status != '000' 은 예외로 올리지 않고 호출자에게 코드로 돌려준다.
    '013'(데이터 없음)은 정상 상황이라 예외로 만들면 배치가 통째로 죽는다.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from xml.etree import ElementTree

import requests

BASE = "https://opendart.fss.or.kr/api"
DAILY_LIMIT = 20_000

# DART 응답 status 코드 중 '정상적인 빈 결과'
EMPTY_STATUSES = {"013"}

REPRT_CODES = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}


# DART 인증키는 40자리 16진수다. 형식 검증이 없으면 잘린 키가 그대로 실려나가
# "등록되지 않은 인증키" 라는, 원인을 짚기 어려운 에러로만 드러난다
# (실측: .env 에 37자로 잘려 저장돼 있었고 그 사실을 알아채는 데 프로브가 필요했다).
_DART_KEY_RE = __import__("re").compile(r"^[0-9a-fA-F]{40}$")


class DartKeyError(RuntimeError):
    """인증키 형식이 DART 규격과 다르다. 호출 전에 잡는다."""


def validate_dart_key(key: str | None, where: str = "") -> str:
    """형식만 본다(유효성은 서버가 판정한다). 값은 절대 메시지에 넣지 않는다."""
    if not key:
        raise DartKeyError(f"DART_API_KEY 가 비어 있다{f' ({where})' if where else ''}")
    if not _DART_KEY_RE.match(key):
        raise DartKeyError(
            f"DART_API_KEY 형식이 잘못됐다{f' ({where})' if where else ''}: "
            f"길이 {len(key)} (40이어야 함), 16진수 여부 "
            f"{bool(__import__('re').match(r'^[0-9a-fA-F]+$', key))}. "
            f"복사 중 잘렸을 가능성이 크다.")
    return key


def _dotenv_key(name: str = "DART_API_KEY") -> str | None:
    """리포 루트의 .env 에서 키를 읽는다(.gitignore 대상).

    의존성 추가 없이 최소한만 파싱한다. 키를 코드나 설정 파일에 두지 않기 위한 장치.
    """
    env = Path(__file__).resolve().parents[3] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class DartError(RuntimeError):
    def __init__(self, status: str, message: str, endpoint: str):
        super().__init__(f"[{endpoint}] DART {status}: {message}")
        self.status, self.message, self.endpoint = status, message, endpoint


@dataclass
class DartClient:
    raw_root: Path
    api_key: str | None = None
    min_interval_s: float = 0.06     # ~16 req/s
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("DART_API_KEY") or _dotenv_key()
        if self.api_key:
            validate_dart_key(self.api_key, "DartClient")
        self.raw_root = Path(self.raw_root)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self._last_call = 0.0
        self._quota_file = self.raw_root / "_quota.json"
        self.session = requests.Session()

    # ── 쿼터 ─────────────────────────────────────────────────────────
    def _quota(self) -> dict:
        today = date.today().isoformat()
        if self._quota_file.exists():
            q = json.loads(self._quota_file.read_text())
            if q.get("date") == today:
                return q
        return {"date": today, "calls": 0}

    def _bump_quota(self) -> None:
        q = self._quota()
        q["calls"] += 1
        self._quota_file.write_text(json.dumps(q))
        if q["calls"] > DAILY_LIMIT:
            raise RuntimeError(f"DART 일일 호출 한도 {DAILY_LIMIT} 초과 — 내일 재개하거나 캐시를 쓸 것")

    def remaining_calls(self) -> int:
        return DAILY_LIMIT - self._quota()["calls"]

    # ── 저수준 호출 + raw 캐시 ────────────────────────────────────────
    def _cache_path(self, endpoint: str, key: str, ext: str = "json") -> Path:
        p = self.raw_root / endpoint / f"{key}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _get_json(self, endpoint: str, params: dict, cache_key: str,
                  refresh: bool = False) -> dict:
        path = self._cache_path(endpoint, cache_key)
        if path.exists() and not refresh:
            return json.loads(path.read_text(encoding="utf-8"))

        if not self.api_key:
            raise RuntimeError(
                f"DART_API_KEY 미설정이고 raw 캐시도 없다: {path}\n"
                f"opendart.fss.or.kr 에서 키를 발급받아 DART_API_KEY 로 설정할 것."
            )

        gap = time.monotonic() - self._last_call
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)

        self._bump_quota()
        r = self.session.get(f"{BASE}/{endpoint}.json",
                             params={**params, "crtfc_key": self.api_key},
                             timeout=self.timeout_s)
        self._last_call = time.monotonic()
        r.raise_for_status()
        payload = r.json()

        status = payload.get("status")
        if status != "000" and status not in EMPTY_STATUSES:
            raise DartError(status, payload.get("message", ""), endpoint)

        # 빈 결과도 캐시한다 — 재실행 시 같은 호출을 반복하지 않기 위해.
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    # ── 엔드포인트 ───────────────────────────────────────────────────
    def corp_codes(self, refresh: bool = False) -> list[dict]:
        """전체 기업 고유번호. corp_code ↔ stock_code 매핑의 유일한 출처.

        zip(xml) 응답이라 별도 처리. 주 1회 갱신이면 충분하다.
        """
        path = self._cache_path("corpCode", "all", ext="xml")
        if not path.exists() or refresh:
            if not self.api_key:
                raise RuntimeError(f"DART_API_KEY 미설정이고 raw 캐시도 없다: {path}")
            self._bump_quota()
            r = self.session.get(f"{BASE}/corpCode.xml",
                                 params={"crtfc_key": self.api_key}, timeout=120)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                path.write_bytes(z.read(z.namelist()[0]))

        root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
        out = []
        for el in root.iter("list"):
            stock = (el.findtext("stock_code") or "").strip()
            if not stock:
                continue          # 비상장 — 스크린 대상 아님
            out.append({
                "corp_code": (el.findtext("corp_code") or "").strip(),
                "ticker": stock,
                "name": (el.findtext("corp_name") or "").strip(),
                "modify_date": (el.findtext("modify_date") or "").strip(),
            })
        return out

    def disclosure_list(self, bgn_de: str, end_de: str, page: int = 1,
                        page_count: int = 100, pblntf_ty: str = "A",
                        refresh: bool = False) -> dict:
        """공시 목록. reported_at(rcept_dt)의 출처이자 PIT 앵커.

        pblntf_ty 는 **인자로 받는다.** 'A'(정기공시)로 하드코딩돼 있어서
        촉매가 사는 곳(B 주요사항보고 / I 거래소공시)에 닿을 수 없었다.
        실측(19일): B 535건, I 3,171건.

        캐시 키에 pblntf_ty 가 들어가야 한다 — 안 넣으면 A 로 받아둔 캐시가
        B 요청에 응답한다. taxonomy['version'] 수기 관리 때와 같은 고장이다.
        """
        return self._get_json(
            "list",
            {"bgn_de": bgn_de, "end_de": end_de, "page_no": page,
             "page_count": page_count, "pblntf_ty": pblntf_ty},
            cache_key=f"{pblntf_ty}_{bgn_de}_{end_de}_p{page}", refresh=refresh)

    def financials(self, corp_code: str, year: int, quarter: int,
                   fs_div: str = "CFS", refresh: bool = False) -> dict:
        """단일회사 전체 재무제표.

        fs_div: CFS(연결) 우선. 없으면 호출자가 OFS 로 재시도한다.
        """
        reprt = REPRT_CODES[quarter]
        return self._get_json(
            "fnlttSinglAcntAll",
            {"corp_code": corp_code, "bsns_year": str(year),
             "reprt_code": reprt, "fs_div": fs_div},
            cache_key=f"{corp_code}_{year}_{reprt}_{fs_div}", refresh=refresh)

    def financials_best_basis(self, corp_code: str, year: int, quarter: int,
                              refresh: bool = False) -> tuple[dict, str]:
        """연결 우선, 없으면 별도. 어느 쪽을 썼는지 반드시 함께 돌려준다.

        지주사가 별도 기준이면 껍데기 숫자다 — 이 구분을 잃으면 PBR 이 통째로 왜곡된다.
        """
        for basis in ("CFS", "OFS"):
            payload = self.financials(corp_code, year, quarter, basis, refresh)
            if payload.get("status") == "000" and payload.get("list"):
                return payload, basis
        return {"status": "013", "list": []}, "NONE"

    def shares_outstanding(self, corp_code: str, year: int, quarter: int,
                           refresh: bool = False) -> dict:
        """주식의 총수 현황. 보통주/우선주/자기주식 분리에 필요."""
        reprt = REPRT_CODES[quarter]
        return self._get_json(
            "stockTotqySttus",
            {"corp_code": corp_code, "bsns_year": str(year), "reprt_code": reprt},
            cache_key=f"{corp_code}_{year}_{reprt}", refresh=refresh)


def parse_rcept_dt(s: str) -> date:
    """DART 날짜는 YYYYMMDD."""
    return datetime.strptime(s.strip(), "%Y%m%d").date()
