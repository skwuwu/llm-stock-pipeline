"""Point-in-time 스토어 (DuckDB).

이 모듈의 존재 이유는 하나다: **룩어헤드를 조심하는 규율이 아니라 데이터 모델로 차단한다.**

모든 재무 fact 는 세 개의 시간 축을 가진다.
  fiscal_end_date  이 숫자가 "언제의" 사실인가        (회계기간 종료일)
  reported_at      우리가 이 숫자를 "언제 알았나"     (DART 접수일 rcept_dt)
  available_at     우리가 이 숫자를 "언제부터 쓸 수 있나" (reported_at + lag)

질의는 항상 available_at <= as_of 로 건다. 정정공시는 기존 행을 UPDATE 하지 않고
revision_of 를 달아 append 한다 — "그때 알던 값"과 "지금 확정된 값"을 둘 다 질의할 수
있어야 백테스트가 정직해진다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

# 흐름(flow) 계정은 기간 누적이라 TTM 환산이 필요하고,
# 잔액(stock) 계정은 시점값이라 그대로 쓴다. 이 구분을 놓치면 PER/PBR이 조용히 틀린다.
FLOW_ELEMENTS = {
    "REVENUE", "OPERATING_INCOME", "NET_INCOME", "NET_INCOME_CONTROLLING",
    "CFO", "CAPEX_PPE", "CAPEX_INTANGIBLE",
}
STOCK_ELEMENTS = {
    "ASSETS", "EQUITY_TOTAL", "EQUITY_CONTROLLING", "CASH_AND_EQUIV",
    "BORROWINGS_SHORT", "BORROWINGS_LONG",
}

DDL = """
CREATE TABLE IF NOT EXISTS facts_financial (
    fact_id           VARCHAR PRIMARY KEY,
    ticker            VARCHAR NOT NULL,
    market            VARCHAR NOT NULL,
    corp_code         VARCHAR,
    fiscal_year       INTEGER NOT NULL,
    fiscal_quarter    INTEGER,          -- 1..4, NULL 이면 연간
    fiscal_end_date   DATE    NOT NULL,
    period_type       VARCHAR NOT NULL, -- FY | CUM (누적) | INSTANT (잔액)
    period_months     INTEGER,          -- CUM 이면 3/6/9/12, INSTANT 면 NULL
    reported_at       DATE    NOT NULL,
    available_at      DATE    NOT NULL,
    statement_basis   VARCHAR NOT NULL, -- CFS | OFS
    element           VARCHAR NOT NULL,
    value             DOUBLE,
    currency          VARCHAR,
    source_doc_id     VARCHAR,          -- rcept_no
    source_url        VARCHAR,
    revision_of       VARCHAR,          -- 정정 대상 rcept_no
    amount_field      VARCHAR,          -- 원문의 어느 필드에서 왔나(thstrm_amount 등).
                                        -- 분기 누적/3개월 필드 해석은 실데이터 검증 전까지
                                        -- 가정이므로, 재수집 없이 감사·정정할 수 있어야 한다.
    ingested_at       TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS prices (
    date              DATE    NOT NULL,
    ticker            VARCHAR NOT NULL,
    close             DOUBLE,
    shares_common     BIGINT,
    shares_preferred  BIGINT,
    treasury_shares   BIGINT,
    market_cap_common DOUBLE,
    market_cap_total  DOUBLE,   -- 보통주 + 우선주. PER 분자는 이 값이어야 한다.
    adtv_20d          DOUBLE,
    -- 교차검증 결과. 소스가 하나면 disagree 는 전부 false 인데, 그건
    -- '이상 없음'이 아니라 '검증하지 못했음'이다(매니페스트로 구별한다).
    price_source           VARCHAR,
    price_source_disagree  BOOLEAN,
    price_alt_close        DOUBLE,
    PRIMARY KEY (date, ticker)
);

CREATE TABLE IF NOT EXISTS security_master (
    ticker         VARCHAR PRIMARY KEY,
    corp_code      VARCHAR,
    name           VARCHAR,
    market         VARCHAR,
    sector_code    VARCHAR,
    listing_date   DATE,
    delisting_date DATE,       -- 생존편향 방지: 폐지 종목도 남긴다
    fiscal_month   INTEGER DEFAULT 12,  -- 결산월. 12 아니면 회계기간 산정에 주의
    is_preferred   BOOLEAN DEFAULT FALSE,
    parent_ticker  VARCHAR,    -- 우선주면 보통주 티커
    is_spac        BOOLEAN DEFAULT FALSE,
    is_reit        BOOLEAN DEFAULT FALSE,
    is_financial   BOOLEAN DEFAULT FALSE,
    is_holding     BOOLEAN DEFAULT FALSE
);

-- 종목 상태(관리종목 등). 소스가 '현재 스냅샷 + 지정일'만 주므로
-- effective_from 으로 PIT 질의는 가능하지만, 과거에 해제된 건은 담기지 않는다.
-- 매일 스냅샷을 쌓으면 released_at 을 직접 관측해 채울 수 있다.
CREATE TABLE IF NOT EXISTS status_events (
    ticker         VARCHAR NOT NULL,
    status         VARCHAR NOT NULL,   -- admin_issue | audit_opinion_bad_admin | suspended
    effective_from DATE,
    reason         VARCHAR,
    source         VARCHAR,
    snapshot_date  DATE    NOT NULL,
    PRIMARY KEY (ticker, status, snapshot_date)
);

-- 공시 목록. 촉매(catalyst)의 앵커다 — rcept_no 없는 촉매는 존재할 수 없다.
--
-- **rcept_dt 가 PIT 앵커다.** 공시 목록은 발표 즉시 공개되므로 재무제표처럼
-- available_lag 를 둘 필요가 없다. as_of 질의는 rcept_dt <= as_of 로 끝난다.
--
-- 정정공시를 **별도 행으로 남긴다.** facts_financial 의 revision_of 와 같은
-- 규율이다 — 원본을 덮어쓰면 '그때 알던 것'을 되물을 수 없다. 정정 여부는
-- is_amendment 로 표시하고, 촉매 검증의 첫 체크(뒤집혔나)가 이걸 읽는다.
CREATE TABLE IF NOT EXISTS disclosures (
    rcept_no     VARCHAR NOT NULL,   -- DART 접수번호. 전역 유일
    ticker       VARCHAR,            -- 비상장 제출인은 NULL 일 수 있다
    corp_code    VARCHAR NOT NULL,
    corp_name    VARCHAR,
    corp_cls     VARCHAR,            -- Y=유가 K=코스닥 N=코넥스 E=기타
    report_nm    VARCHAR NOT NULL,   -- 원문 그대로. 정규화본은 report_key
    report_key   VARCHAR NOT NULL,   -- 정정 접두어 제거본. 패턴 매칭 대상
    is_amendment BOOLEAN NOT NULL,
    pblntf_ty    VARCHAR NOT NULL,   -- A정기 B주요사항 I거래소 …
    rcept_dt     DATE    NOT NULL,
    flr_nm       VARCHAR,
    ingested_at  TIMESTAMP DEFAULT now(),
    PRIMARY KEY (rcept_no)
);

CREATE INDEX IF NOT EXISTS idx_facts_lookup
    ON facts_financial (ticker, element, available_at);
CREATE INDEX IF NOT EXISTS idx_disc_lookup
    ON disclosures (ticker, rcept_dt);
"""


def make_fact_id(ticker: str, element: str, fiscal_end_date: date,
                 period_type: str, statement_basis: str, source_doc_id: str) -> str:
    """source_doc_id 를 포함하므로 정정공시는 새 행이 된다(덮어쓰지 않는다)."""
    h = hashlib.sha256()
    for p in (ticker, element, str(fiscal_end_date), period_type, statement_basis, source_doc_id or ""):
        h.update(p.encode())
        h.update(b"\x00")
    return h.hexdigest()[:32]


@dataclass
class PitStore:
    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.path))
        self.con.execute(DDL)

    def close(self) -> None:
        self.con.close()

    # ── 적재 ─────────────────────────────────────────────────────────
    def append_facts(self, df: pd.DataFrame) -> int:
        """fact 행을 append. fact_id 중복은 무시(멱등 재실행)."""
        if df.empty:
            return 0
        self.con.register("_incoming", df)
        before = self.con.execute("SELECT count(*) FROM facts_financial").fetchone()[0]
        self.con.execute("""
            INSERT INTO facts_financial BY NAME
            SELECT * FROM _incoming
            WHERE fact_id NOT IN (SELECT fact_id FROM facts_financial)
        """)
        self.con.unregister("_incoming")
        after = self.con.execute("SELECT count(*) FROM facts_financial").fetchone()[0]
        return after - before

    def upsert_prices(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.con.register("_p", df)
        self.con.execute("DELETE FROM prices WHERE (date, ticker) IN (SELECT date, ticker FROM _p)")
        self.con.execute("INSERT INTO prices BY NAME SELECT * FROM _p")
        self.con.unregister("_p")
        return len(df)

    def upsert_master(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.con.register("_m", df)
        self.con.execute("DELETE FROM security_master WHERE ticker IN (SELECT ticker FROM _m)")
        self.con.execute("INSERT INTO security_master BY NAME SELECT * FROM _m")
        self.con.unregister("_m")
        return len(df)

    # ── PIT 질의 ─────────────────────────────────────────────────────
    def facts_asof(self, as_of: date, tickers: list[str] | None = None) -> pd.DataFrame:
        """as_of 시점에 알 수 있었던 모든 fact.

        정정공시가 있으면 같은 (ticker, element, 기간)에 여러 행이 존재하므로,
        reported_at 이 가장 늦은 행을 채택한다 — 단, available_at <= as_of 범위 안에서만.
        미래의 정정을 끌어오지 않는 것이 핵심.
        """
        where_t = "AND ticker = ANY(?)" if tickers else ""
        params: list = [as_of] + ([tickers] if tickers else [])
        return self.con.execute(f"""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY ticker, element, fiscal_end_date, period_type, statement_basis
                    ORDER BY reported_at DESC, ingested_at DESC
                ) AS rn
                FROM facts_financial
                WHERE available_at <= ? {where_t}
            ) WHERE rn = 1
        """, params).df()

    def latest_fact_asof(self, as_of: date, element: str,
                         tickers: list[str] | None = None) -> pd.DataFrame:
        """종목별로 as_of 시점 가장 최근 회계기간의 element 값 하나."""
        df = self.facts_asof(as_of, tickers)
        if df.empty:
            return df
        df = df[df["element"] == element]
        if df.empty:
            return df
        # 연결(CFS) 우선, 없으면 별도(OFS). 어느 쪽을 썼는지는 컬럼으로 남는다.
        df = df.assign(_basis_rank=(df["statement_basis"] != "CFS").astype(int))
        df = df.sort_values(["ticker", "_basis_rank", "fiscal_end_date", "reported_at"],
                            ascending=[True, True, False, False])
        return df.groupby("ticker", as_index=False).first().drop(columns=["_basis_rank"])

    def prices_asof(self, as_of: date) -> pd.DataFrame:
        """as_of 이전 마지막 거래일의 시세 (휴장일 대응)."""
        return self.con.execute("""
            SELECT * EXCLUDE (rn) FROM (
                SELECT *, row_number() OVER (PARTITION BY ticker ORDER BY date DESC) rn
                FROM prices WHERE date <= ?
            ) WHERE rn = 1
        """, [as_of]).df()

    def master(self) -> pd.DataFrame:
        return self.con.execute("SELECT * FROM security_master").df()

    def upsert_status(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.con.register("_s", df)
        self.con.execute("""
            DELETE FROM status_events
            WHERE (ticker, status, snapshot_date) IN
                  (SELECT ticker, status, snapshot_date FROM _s)
        """)
        self.con.execute("INSERT INTO status_events BY NAME SELECT * FROM _s")
        self.con.unregister("_s")
        return len(df)

    def status_asof(self, as_of: date) -> pd.DataFrame:
        """as_of 시점에 유효한 상태.

        effective_from <= as_of 인 것만 본다. snapshot_date > as_of 인 스냅샷도
        쓰는데, 그 안의 effective_from 은 과거 사실이므로 룩어헤드가 아니다.
        다만 '그 사이 해제된 종목'은 알 수 없다 — 과소 적용(누락) 방향의 편의다.
        """
        return self.con.execute("""
            SELECT ticker, status, min(effective_from) AS effective_from,
                   any_value(reason) AS reason, max(snapshot_date) AS snapshot_date
            FROM status_events
            WHERE effective_from IS NOT NULL AND effective_from <= ?
            GROUP BY ticker, status
        """, [as_of]).df()

    def upsert_disclosures(self, df: pd.DataFrame) -> int:
        """rcept_no 는 전역 유일이므로 재수집해도 중복되지 않는다."""
        if df.empty:
            return 0
        before = self.con.execute("SELECT count(*) FROM disclosures").fetchone()[0]
        self.con.register("_d", df)
        self.con.execute("""
            INSERT INTO disclosures BY NAME
            SELECT * FROM _d WHERE rcept_no NOT IN (SELECT rcept_no FROM disclosures)
        """)
        self.con.unregister("_d")
        after = self.con.execute("SELECT count(*) FROM disclosures").fetchone()[0]
        return after - before

    def disclosures_asof(self, as_of: date, since: date | None = None,
                         tickers: list[str] | None = None) -> pd.DataFrame:
        """as_of 시점에 알 수 있었던 공시.

        공시 목록은 발표 즉시 공개되므로 available_lag 가 없다 —
        rcept_dt <= as_of 가 곧 PIT 경계다.
        """
        w = ["rcept_dt <= ?"]
        params: list = [as_of]
        if since is not None:
            w.append("rcept_dt >= ?")
            params.append(since)
        if tickers:
            w.append("ticker = ANY(?)")
            params.append(tickers)
        return self.con.execute(
            f"SELECT * FROM disclosures WHERE {' AND '.join(w)} ORDER BY rcept_dt DESC",
            params).df()

    def revision_history(self, ticker: str, element: str) -> pd.DataFrame:
        """같은 회계기간에 대해 값이 어떻게 정정돼 왔는지. 감사용."""
        return self.con.execute("""
            SELECT fiscal_end_date, reported_at, value, source_doc_id, revision_of
            FROM facts_financial
            WHERE ticker = ? AND element = ?
            ORDER BY fiscal_end_date, reported_at
        """, [ticker, element]).df()
