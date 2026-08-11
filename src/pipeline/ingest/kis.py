"""한국투자증권 KIS Open API 가격 소스.

엔드포인트·tr_id·파라미터는 **공식 저장소에서 대조 확인**했다(2026-08-07):
    github.com/koreainvestment/open-trading-api
    examples_llm/domestic_stock/{inquire_price, intstock_multprice}

⚠ 다만 **응답 필드명과 레이트 리밋은 여전히 미검증이다.** 계정 자격증명이 없어
실제 응답을 본 적이 없다. **반드시 프로브로 확인한 뒤** 대량 수집에 쓸 것:

    pipeline probe-price --source kis --as-of <date>

프로브를 통과하지 못하면 `ingest-prices --source kis` 는 거부된다. 추측한 배선으로
2,598종목을 돌리다 실패하면, 실패를 알아채는 시점이 레이트 리밋을 다 쓴 뒤가 된다.

자격증명(.env):
    KIS_APP_KEY=...
    KIS_APP_SECRET=...
    KIS_ENV=real            # real | vps(모의)

공식 저장소는 pip 패키지가 아니라 **샘플 코드**다. 의존성으로 쓰지 않고 사양 확인에만
썼다. 그쪽 설정 파일(kis_devlp.yaml)은 계좌번호까지 요구하는데, 우리는 시세만 읽으므로
**계좌번호를 저장하지 않는다** — 주문 권한이 필요 없는 자격증명만 둔다.

메모 — 프로브로 확인할 것:
  · 응답 필드명(stck_prpr, acml_tr_pbmn 등)이 실제와 맞는가
  · 초당/일별 레이트 리밋 (문서상 20건/초로 알려져 있으나 미확인)
  · 모의투자(vps)에서도 시세 조회가 되는가 — tr_id 는 실전과 같다고 돼 있다
  · 20일 거래대금은 없다. 당일 누적 거래대금만 온다 →
    inquire_daily_itemchartprice(일봉)로 따로 받거나 자체 이력에서 계산해야 한다
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from pipeline.ingest.price_sources import OUT_COLUMNS, PriceSourceError

HOSTS = {"real": "https://openapi.koreainvestment.com:9443",
         "vps": "https://openapivts.koreainvestment.com:29443"}
TOKEN_PATH = "/oauth2/tokenP"
# 단건 조회. 프로브용 — 대량에는 쓰지 않는다.
QUOTE_PATH = "/uapi/domestic-stock/v1/quotations/inquire-price"
QUOTE_TR = "FHKST01010100"          # 실전·모의 동일
# 복수종목 조회. **한 번에 30종목** — 2,598종목이 87콜로 끝난다.
# 단건으로 돌면 2,598콜이라 일별 운영이 성립하지 않는다.
MULTI_PATH = "/uapi/domestic-stock/v1/quotations/intstock-multprice"
MULTI_TR = "FHKST11300006"
MULTI_BATCH = 30
# 초당 호출 제한(문서상 20건/초). 여유를 두고 잡는다 — 막히면 토큰까지 잠긴다.
MIN_INTERVAL_S = 0.08


class KisNotVerifiedError(PriceSourceError):
    """프로브를 통과하지 않은 소스로 대량 수집을 시도했다."""


def _cred(name: str) -> str | None:
    v = os.environ.get(name)
    if v:
        return v
    env = Path(__file__).resolve().parents[3] / ".env"
    if not env.exists():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(f"{name}=") and not line.startswith("#"):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


@dataclass
class KisSource:
    cache_root: Path
    env: str = field(default_factory=lambda: _cred("KIS_ENV") or "real")
    name: str = "kis"
    verified: bool = False          # probe 를 통과해야 True (마커 파일로 유지)

    def __post_init__(self) -> None:
        self.app_key = _cred("KIS_APP_KEY")
        self.app_secret = _cred("KIS_APP_SECRET")
        if self.env not in HOSTS:
            raise PriceSourceError(f"KIS_ENV 는 {list(HOSTS)} 중 하나여야 한다: {self.env!r}")
        self.cache_root = Path(self.cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self._token: str | None = None
        self._token_exp = datetime.min
        self._last = 0.0
        self.calls = 0
        # 검증 상태를 파일로 남긴다 — 프로브와 수집이 다른 프로세스라
        # 메모리 플래그만으로는 게이트가 성립하지 않는다.
        self._mark = self.cache_root / f"_verified_{self.env}.json"
        if self._mark.exists():
            self.verified = True
            self.verified_at = json.loads(self._mark.read_text(encoding="utf-8"))
        else:
            self.verified_at = None

    def mark_verified(self, as_of: date, n: int) -> None:
        self._mark.write_text(json.dumps(
            {"probed_as_of": str(as_of), "tickers_ok": n,
             "at": datetime.now().isoformat(timespec="seconds"), "env": self.env},
            ensure_ascii=False), encoding="utf-8")
        self.verified = True

    # ── 인증 ─────────────────────────────────────────────────────
    def _require_creds(self) -> None:
        missing = [n for n, v in (("KIS_APP_KEY", self.app_key),
                                  ("KIS_APP_SECRET", self.app_secret)) if not v]
        if missing:
            raise PriceSourceError(
                f"KIS 자격증명 없음: {missing}. .env 에 추가할 것 — "
                f"발급은 한국투자증권 KIS Developers 에서 한다.")

    def token(self) -> str:
        """접근토큰. 문서상 24시간 유효라 캐시해 재사용한다 —
        매 호출 재발급하면 발급 자체가 레이트 리밋에 걸린다."""
        self._require_creds()
        if self._token and datetime.now() < self._token_exp:
            return self._token
        cache = self.cache_root / f"_token_{self.env}.json"
        if cache.exists():
            d = json.loads(cache.read_text(encoding="utf-8"))
            exp = datetime.fromisoformat(d["expires_at"])
            if datetime.now() < exp:
                self._token, self._token_exp = d["token"], exp
                return self._token

        r = self.session.post(
            HOSTS[self.env] + TOKEN_PATH,
            json={"grant_type": "client_credentials", "appkey": self.app_key,
                  "appsecret": self.app_secret}, timeout=20)
        if r.status_code != 200:
            raise PriceSourceError(f"KIS 토큰 발급 실패 {r.status_code}: {r.text[:200]}")
        d = r.json()
        tok = d.get("access_token")
        if not tok:
            raise PriceSourceError(f"KIS 응답에 access_token 이 없다: {list(d)}")
        # 만료 여유 10분
        self._token_exp = datetime.now() + timedelta(
            seconds=int(d.get("expires_in", 86400)) - 600)
        self._token = tok
        cache.write_text(json.dumps({"token": tok,
                                     "expires_at": self._token_exp.isoformat()}),
                         encoding="utf-8")
        return tok

    # ── 시세 ─────────────────────────────────────────────────────
    def _throttle(self) -> None:
        gap = time.monotonic() - self._last
        if gap < MIN_INTERVAL_S:
            time.sleep(MIN_INTERVAL_S - gap)
        self._last = time.monotonic()

    def quote(self, ticker: str) -> dict:
        """종목 현재가. 응답 필드명은 미검증 — 프로브로 확인할 것."""
        self._throttle()
        r = self.session.get(
            HOSTS[self.env] + QUOTE_PATH,
            headers={"authorization": f"Bearer {self.token()}",
                     "appkey": self.app_key, "appsecret": self.app_secret,
                     "tr_id": QUOTE_TR},
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
            timeout=20)
        self.calls += 1
        if r.status_code != 200:
            raise PriceSourceError(f"{ticker}: KIS {r.status_code} {r.text[:150]}")
        d = r.json()
        if d.get("rt_cd") not in ("0", 0):
            raise PriceSourceError(f"{ticker}: rt_cd={d.get('rt_cd')} {d.get('msg1')}")
        return d.get("output") or {}

    def fetch(self, tickers: list[str], as_of: date,
              refresh: bool = False) -> pd.DataFrame:
        if not self.verified:
            raise KisNotVerifiedError(
                "KIS 소스는 프로브를 통과하지 않았다. "
                "`pipeline probe-price --source kis` 로 먼저 확인할 것 — "
                "엔드포인트·필드명이 미검증이라 대량 수집 중 실패하면 "
                "레이트 리밋을 소진한 뒤에야 알게 된다.")
        # 인증은 루프 **밖에서** 한 번 확인한다. 종목별 except 로 삼키면
        # '자격증명 없음' 이 '종가를 하나도 못 받음' 으로 둔갑해 원인이 감춰진다.
        self.token()
        # 30종목씩 묶어 받는다. 단건으로 돌면 2,598콜이라 일별 운영이 성립하지 않는다.
        if len(tickers) > MULTI_BATCH:
            return self._fetch_multi(tickers, as_of)
        rows = []
        for t in tickers:
            try:
                o = self.quote(t)
            except PriceSourceError:
                continue      # 개별 종목 실패만 넘긴다(상장폐지·거래정지 등)
            close = _num(o.get("stck_prpr"))          # 주식 현재가
            amt = _num(o.get("acml_tr_pbmn"))         # 누적 거래대금(당일)
            if close is None:
                continue
            rows.append({"ticker": t, "close": close,
                         "adtv_20d": amt, "price_date": as_of})
        return pd.DataFrame(rows).reindex(columns=OUT_COLUMNS)


    def _fetch_multi(self, tickers: list[str], as_of: date) -> pd.DataFrame:
        """복수종목 시세. 30종목/콜 — 2,598종목이 87콜로 끝난다.

        응답이 요청 순서와 같다는 보장은 확인하지 못했다. 그래서 응답에 종목코드가
        실려 오면 그것으로 맞추고, 없으면 **그 배치를 통째로 버린다.**
        순서를 가정하고 잘못 매칭하면 A 종목 가격이 B 종목에 붙는데,
        그건 데이터가 없는 것보다 훨씬 나쁘다.
        """
        rows: list[dict] = []
        for i in range(0, len(tickers), MULTI_BATCH):
            batch = tickers[i:i + MULTI_BATCH]
            params: dict[str, str] = {}
            for n, t in enumerate(batch, start=1):
                params[f"FID_COND_MRKT_DIV_CODE_{n}"] = "J"
                params[f"FID_INPUT_ISCD_{n}"] = t
            self._throttle()
            r = self.session.get(
                HOSTS[self.env] + MULTI_PATH,
                headers={"authorization": f"Bearer {self.token()}",
                         "appkey": self.app_key, "appsecret": self.app_secret,
                         "tr_id": MULTI_TR},
                params=params, timeout=30)
            self.calls += 1
            if r.status_code != 200:
                continue
            d = r.json()
            if d.get("rt_cd") not in ("0", 0):
                continue
            for o in (d.get("output") or []):
                code = (o.get("inter_shrn_iscd") or o.get("mksc_shrn_iscd")
                        or o.get("stck_shrn_iscd"))
                close = _num(o.get("inter2_prpr") or o.get("stck_prpr"))
                if not code or close is None:
                    continue        # 종목코드를 못 찾으면 매칭할 수 없다 — 버린다
                rows.append({"ticker": str(code).zfill(6), "close": close,
                             "adtv_20d": _num(o.get("acml_tr_pbmn")),
                             "price_date": as_of})
        return pd.DataFrame(rows).reindex(columns=OUT_COLUMNS)


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None
