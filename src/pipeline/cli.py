"""파이프라인 CLI. 각 커맨드는 멱등이고, 완료는 산출물 경로로 판정한다.

    python -m pipeline.cli init
    python -m pipeline.cli ingest-master
    python -m pipeline.cli ingest-dart   --year 2024 --quarter 4 --limit 50
    python -m pipeline.cli ingest-prices --as-of 2025-04-01 [--csv path]
    python -m pipeline.cli derive        --as-of 2025-04-01
    python -m pipeline.cli screen        --as-of 2025-04-01 --target 60
    python -m pipeline.cli status
"""

from __future__ import annotations

import argparse
import json
import sys

# Windows 콘솔 기본 인코딩(cp949)에서 일부 문자가 죽어 배치가 중단된다.
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from pipeline.derive.metrics import (assert_metrics_sane, build_metrics,
                                     data_defect_rate)
from pipeline.ingest.dart import DartClient
from pipeline.ingest.kind import KindClient
from pipeline.ingest.krx import infer_parent_ticker, load_csv
from pipeline.ingest.prices import PriceIngestor, build_price_table
from pipeline.normalize.kr import normalize_financials
from pipeline.normalize.sector import annotate, coverage_report
from pipeline.screen.gate import (apply_hard_guards, filters_from_config,
                                  guard_effectiveness, run_gate)
from pipeline.store.pit import PitStore

REPO = Path(__file__).resolve().parents[2]


DATA = REPO / "data"
DB = DATA / "pit.duckdb"
RAW_DART = DATA / "raw" / "dart"


def _iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _store() -> PitStore:
    return PitStore(DB)


# ── 커맨드 ───────────────────────────────────────────────────────────
def cmd_init(_a) -> int:
    for p in ["raw/dart", "raw/krx", "normalized", "derived", "screens", "enrich",
              "llm/tags", "verify", "out"]:
        (DATA / p).mkdir(parents=True, exist_ok=True)
    s = _store()
    print(f"PIT 스토어 초기화: {DB}")
    print(s.con.execute("SELECT table_name FROM duckdb_tables()").df().to_string(index=False))
    s.close()
    return 0


def cmd_ingest_master(a) -> int:
    """KIND 로 상장사 마스터를 세운다. 업종·결산월 포함, 인증 불필요.

    DART corp_code 는 선택 — 키가 있으면 붙이고, 없으면 비운다(재무 수집 시 필요).
    """
    kind = KindClient(DATA / "raw" / "kind")
    df = annotate(kind.corp_list(refresh=a.refresh))
    rep = coverage_report(df)

    df["parent_ticker"] = df["ticker"].map(infer_parent_ticker)
    df["is_preferred"] = df["parent_ticker"].notna()
    df["delisting_date"] = None
    df["corp_code"] = None

    if a.with_dart:
        try:
            codes = pd.DataFrame(DartClient(RAW_DART).corp_codes())
            df = df.drop(columns=["corp_code"]).merge(
                codes[["ticker", "corp_code"]], on="ticker", how="left")
            print(f"DART corp_code 매칭 {int(df['corp_code'].notna().sum())}/{len(df)}")
        except Exception as e:                                   # noqa: BLE001
            print(f"DART corp_code 생략: {e}", file=sys.stderr)

    cols = ["ticker", "corp_code", "name", "market", "sector_code", "listing_date",
            "delisting_date", "fiscal_month", "is_preferred", "parent_ticker",
            "is_spac", "is_reit", "is_financial", "is_holding"]
    s = _store()
    n = s.upsert_master(df.reindex(columns=cols))

    # 관리종목 상태
    adm = kind.admin_issues(refresh=a.refresh)
    events = [adm.assign(status="admin_issue", source="kind")[
        ["ticker", "status", "designation_date", "reason", "source", "snapshot_date"]]]
    audit = adm[adm["audit_opinion_bad"]]
    if not audit.empty:
        events.append(audit.assign(status="audit_opinion_bad_admin", source="kind")[
            ["ticker", "status", "designation_date", "reason", "source", "snapshot_date"]])
    ev = pd.concat(events, ignore_index=True).rename(
        columns={"designation_date": "effective_from"})
    ns = s.upsert_status(ev)
    s.close()

    print(f"security_master {n}종목 | 섹터 매핑 {rep['coverage']:.2%} "
          f"({rep['unmapped_tickers']}종목 미매핑)")
    print(f"결산월 12월 아님 {int((df['fiscal_month'] != 12).sum())} | "
          f"스팩 {int(df['is_spac'].sum())} | 리츠 {int(df['is_reit'].sum())} | "
          f"금융 {int(df['is_financial'].sum())} | 지주(휴리스틱) {int(df['is_holding'].sum())}")
    print(f"status_events {ns}건 (관리종목 {len(adm)}, 감사의견 {int(adm['audit_opinion_bad'].sum())})")
    if rep["unmapped_industries"]:
        print(f"미매핑 업종: {rep['unmapped_industries']}", file=sys.stderr)
    print("주의: 관리종목은 현재 스냅샷 + 지정일이다. 과거에 해제된 건은 담기지 않아 "
          "과거 as_of 질의에서 과소 적용된다.")
    return 0


def cmd_ingest_dart(a) -> int:
    s = _store()
    master = s.master()
    if master.empty:
        print("security_master 가 비었다. 먼저 ingest-master 를 실행할 것.", file=sys.stderr)
        return 1

    targets = master[(~master["is_preferred"].fillna(False))
                     & (~master["is_spac"].fillna(False))]
    if a.tickers:
        targets = targets[targets["ticker"].isin(a.tickers)]
    if a.listed_before:
        # KIND 목록은 상장일 역순이라 head(N) 은 '최근 IPO N개'가 된다.
        # 공시 이력이 없는 코호트를 뽑아놓고 커버리지를 논하면 의미가 없다.
        targets = targets[pd.to_datetime(targets["listing_date"], errors="coerce")
                          < pd.Timestamp(a.listed_before)]
    if a.sample:
        targets = targets.sample(min(a.sample, len(targets)), random_state=a.seed)
    elif a.limit:
        targets = targets.head(a.limit)

    client = DartClient(RAW_DART)
    total, ok, empty, failed = 0, 0, 0, 0
    for row in targets.itertuples():
        try:
            payload, basis = client.financials_best_basis(row.corp_code, a.year, a.quarter)
        except Exception as e:                                  # noqa: BLE001
            failed += 1
            print(f"  {row.ticker} 실패: {e}", file=sys.stderr)
            continue
        if basis == "NONE":
            empty += 1
            continue
        df = normalize_financials(
            payload, ticker=row.ticker, statement_basis=basis,
            available_lag_days=a.lag,
            fiscal_month=int(row.fiscal_month or 12),
        )
        total += s.append_facts(df)
        ok += 1

    s.close()
    print(f"대상 {len(targets)} | 적재 {ok} (신규 fact {total}) | 데이터없음 {empty} | 실패 {failed}")
    print(f"DART 잔여 호출 {client.remaining_calls():,}")
    return 0


def cmd_renormalize(a) -> int:
    """캐시된 raw 응답만으로 facts 를 통째로 재생성한다. API 호출 0회.

    raw 를 불변으로 보존하는 이유가 이것이다 — 파서 가정이 틀렸을 때
    재수집 없이 고칠 수 있어야 한다.
    """
    import json as _json
    s = _store()
    master = s.master()
    fm = dict(zip(master["ticker"], master["fiscal_month"].fillna(12).astype(int)))
    by_corp = {c: t for t, c in zip(master["ticker"], master["corp_code"]) if c}

    root = RAW_DART / "fnlttSinglAcntAll"
    files = sorted(root.glob("*.json"))
    if not files:
        print(f"{root} 에 캐시된 raw 가 없다.", file=sys.stderr)
        return 1

    before = s.con.execute("SELECT count(*) FROM facts_financial").fetchone()[0]
    s.con.execute("DELETE FROM facts_financial")

    total, skipped = 0, 0
    for fp in files:
        corp, _year, _reprt, basis = fp.stem.split("_")
        ticker = by_corp.get(corp)
        if ticker is None:
            skipped += 1
            continue
        payload = _json.loads(fp.read_text(encoding="utf-8"))
        if payload.get("status") != "000":
            continue
        df = normalize_financials(payload, ticker=ticker, statement_basis=basis,
                                  available_lag_days=a.lag,
                                  fiscal_month=fm.get(ticker, 12))
        total += s.append_facts(df)

    s.close()
    print(f"raw {len(files)}건 재파싱 | facts {before:,} → {total:,} "
          f"(변화 {total - before:+,}) | 매핑 실패 {skipped}")
    return 0


def cmd_ingest_prices(a) -> int:
    """FDR 종가 × DART 주식수 → 시총. pykrx 는 KRX 로그인 요구로 사용 불가."""
    s = _store()
    master = s.master()
    if a.csv:
        n = s.upsert_prices(load_csv(Path(a.csv), master))
        s.close()
        print(f"prices {n}행 @ {a.as_of} (CSV)")
        return 0

    pool = master[(~master["is_preferred"].fillna(False))
                  & (~master["is_spac"].fillna(False))]
    if a.tickers:
        pool = pool[pool["ticker"].isin(a.tickers)]
    if a.with_facts:
        # 재무가 적재된 종목만 — 시세만 있고 재무가 없으면 지표를 못 만든다
        have = set(s.con.execute(
            "SELECT DISTINCT ticker FROM facts_financial").df()["ticker"])
        pool = pool[pool["ticker"].isin(have)]
    if a.limit:
        pool = pool.head(a.limit)

    from pipeline.ingest.price_sources import reconcile

    # 소스를 여럿 받으면 종가를 대조한다. 단일 소스로는 주가 오류를 잡을 수 없다 —
    # 시총·PER·PBR 이 전부 같은 종가에서 나와 서로를 검증하지 못한다.
    tickers = pool["ticker"].tolist()
    frames: dict[str, pd.DataFrame] = {}
    for name in dict.fromkeys([a.source, *(a.cross_check or [])]):
        src = _price_source(name)
        frames[name] = src.fetch(tickers, a.as_of, refresh=a.refresh)
        print(f"  [{name}] {int(frames[name]['close'].notna().sum())}/{len(tickers)}종목")
    rec = reconcile(frames, primary=a.source)
    px = rec.frame
    if rec.compared_with:
        print(f"교차검증 {rec.n_compared}종목 대조 → 불일치 {rec.n_disagree}건 "
              f"(primary={rec.primary}, 비교={rec.compared_with})")
    else:
        # '검증했는데 이상 없음'이 아니라 '검증하지 못했음'이다. 구별해서 말한다.
        print(f"교차검증 없음 — 소스가 {a.source} 하나뿐이라 주가 오류를 잡을 수 없다. "
              f"--cross-check 로 두 번째 소스를 붙일 것.", file=sys.stderr)
    print(f"시세 {int(px['close'].notna().sum())}/{len(pool)}종목")

    ing = PriceIngestor(DATA / "raw" / "prices", workers=a.workers)

    corp_map = dict(zip(pool["ticker"], pool["corp_code"]))
    sh = ing.fetch_shares(DartClient(RAW_DART), corp_map, a.shares_year, a.shares_quarter)
    print(f"주식수 {len(sh)}/{len(corp_map)}종목")

    tbl = build_price_table(px, sh, a.as_of)
    for c in ("price_source", "price_source_disagree", "price_alt_close"):
        if c in px.columns:
            tbl[c] = px.set_index("ticker")[c].reindex(tbl["ticker"]).values
    missing_cap = int(tbl["market_cap_total"].isna().sum())
    n = s.upsert_prices(tbl)
    s.close()
    print(f"prices {n}행 @ {a.as_of} | 시총 결측 {missing_cap} (하드가드에서 배제됨)")
    if not tbl.empty:
        top = tbl.nlargest(5, "market_cap_total")[["ticker", "close", "market_cap_total"]]
        top = top.merge(master[["ticker", "name"]], on="ticker", how="left")
        print("시총 상위 5 (스케일 눈검증용):")
        for r in top.itertuples():
            print(f"  {r.ticker} {r.name or '':<12} {r.close:>10,.0f}원  "
                  f"{r.market_cap_total/1e12:>8,.1f}조")
    return 0


def cmd_derive(a) -> int:
    s = _store()
    facts = s.facts_asof(a.as_of)
    prices = s.prices_asof(a.as_of)
    if facts.empty or prices.empty:
        print(f"facts {len(facts)} / prices {len(prices)} — 데이터 부족", file=sys.stderr)
        s.close()
        return 1

    m = build_metrics(facts, prices, s.master(), a.as_of, status=s.status_asof(a.as_of))
    out = DATA / "derived" / f"metrics_{a.as_of}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(out, index=False)

    # 계산은 도는데 결과가 전부 NaN 인 고장은 예외를 던지지 않는다.
    # 산출물 자체를 검사해야 잡힌다(실측: _safe_div 스칼라 분모 버그).
    kpi = data_defect_rate(m)
    kpi["nan_rate"] = assert_metrics_sane(m)
    (DATA / "derived" / f"quality_{a.as_of}.json").write_text(
        json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    s.close()

    print(f"metrics → {out}  ({len(m)}종목)")
    print(json.dumps(kpi, ensure_ascii=False, indent=2))
    return 0


def cmd_ingest_holder(a) -> int:
    """DART hyslrSttus → facts. 종목당 1콜. 배당 수집과 같은 PIT 규율."""
    from pipeline.ingest.dart_holder import HolderClient, HolderFetchError, parse_holders
    return _ingest_annual(a, HolderClient(DATA / "raw" / "dart" / "hyslrSttus"),
                          parse_holders, HolderFetchError, "최대주주")


def _ingest_annual(a, cli, parse, err_cls, label: str) -> int:
    """(corp, year) 단위 연간 보고 항목 공통 수집 루틴.

    수집 실패와 '데이터 없음' 을 구분해 집계한다 — 미수집을 0 으로 두면
    그 종목이 관련 스크린에서 조용히 탈락한다.
    """
    from pipeline.store.pit import make_fact_id
    s = _store()
    master = s.con.execute(
        "SELECT ticker, corp_code FROM security_master WHERE corp_code IS NOT NULL").df()
    if a.tickers:
        master = master[master["ticker"].isin(a.tickers)]
    if a.limit:
        master = master.head(a.limit)

    facts, no_data, failed = [], [], []
    for i, r in enumerate(master.itertuples(), 1):
        try:
            rows = parse(cli.fetch(r.corp_code, a.year, refresh=a.refresh),
                         r.ticker, lag_days=a.lag)
        except (err_cls, requests.RequestException) as e:
            failed.append((r.ticker, str(e)[:80]))
            continue
        (facts.extend(rows) if rows else no_data.append(r.ticker))
        if i % 400 == 0:
            print(f"  {i}/{len(master)} (API {cli.calls}콜)", file=sys.stderr)

    if facts:
        df = pd.DataFrame(facts)
        df["market"] = "KR"
        df["statement_basis"] = "DART_ANNUAL"
        df["currency"] = "KRW"
        df["fiscal_year"] = a.year
        df["fiscal_quarter"] = 4
        df["period_type"] = "FY"
        df["period_months"] = 12
        df["amount_field"] = "thstrm"
        df["source_url"] = ("https://dart.fss.or.kr/dsaf001/main.do?rcpNo="
                            + df["source_doc_id"])
        df["fact_id"] = [make_fact_id(x.ticker, x.element, x.fiscal_end_date,
                                      x.period_type, x.statement_basis, x.source_doc_id)
                         for x in df.itertuples()]
        s.append_facts(df)

    print(f"{label} 수집 {len(master)}종목 | facts {len(facts)}건 | "
          f"데이터 없음 {len(no_data)} | 실패 {len(failed)} | API {cli.calls}콜")
    if failed:
        print(f"  실패 예시: {failed[:3]}", file=sys.stderr)
    return 0


def cmd_ingest_dividend(a) -> int:
    """DART alotMatter → facts. 종목당 1콜.

    수집 실패와 무배당을 구분해 집계한다 — 미수집을 무배당으로 두면
    그 종목이 배당 스크린에서 조용히 탈락한다.
    """
    from pipeline.ingest.dart_dividend import (DividendClient, DividendFetchError,
                                               parse_dividend)
    s = _store()
    master = s.con.execute(
        "SELECT ticker, name, corp_code FROM security_master WHERE corp_code IS NOT NULL"
    ).df()
    if a.tickers:
        master = master[master["ticker"].isin(a.tickers)]
    if a.limit:
        master = master.head(a.limit)

    cli = DividendClient(DATA / "raw" / "dart" / "alotMatter")
    facts, no_data, failed = [], [], []
    for i, r in enumerate(master.itertuples(), 1):
        try:
            raw = cli.fetch(r.corp_code, a.year, refresh=a.refresh)
            rows = parse_dividend(raw, r.ticker, lag_days=a.lag)
        except (DividendFetchError, requests.RequestException) as e:
            failed.append((r.ticker, str(e)[:80]))
            continue
        if rows:
            facts.extend(rows)
        else:
            no_data.append(r.ticker)
        if i % 200 == 0:
            print(f"  {i}/{len(master)} (API {cli.calls}콜)", file=sys.stderr)

    if facts:
        from pipeline.store.pit import make_fact_id
        df = pd.DataFrame(facts)
        df["market"] = "KR"
        df["statement_basis"] = "DART_ALOT"
        df["currency"] = "KRW"
        df["fiscal_year"] = a.year
        df["fiscal_quarter"] = 4
        # 배당은 그 사업연도 전체에 대해 확정되는 값이라 연간(FY)이다.
        df["period_type"] = "FY"
        df["period_months"] = 12
        df["amount_field"] = "thstrm"
        df["source_url"] = ("https://dart.fss.or.kr/dsaf001/main.do?rcpNo="
                            + df["source_doc_id"])
        # source_doc_id 를 포함하므로 정정공시는 덮어쓰지 않고 새 행이 된다.
        df["fact_id"] = [
            make_fact_id(r.ticker, r.element, r.fiscal_end_date, r.period_type,
                         r.statement_basis, r.source_doc_id)
            for r in df.itertuples()]
        s.append_facts(df)

    print(f"배당 수집 {len(master)}종목 | facts {len(facts)}건 | "
          f"데이터 없음 {len(no_data)} | 실패 {len(failed)} | API {cli.calls}콜")
    if failed:
        print(f"  실패 예시: {failed[:3]}", file=sys.stderr)
    return 0


def _price_source(name: str):
    """이름 → 가격 소스. 미검증 소스는 fetch 단계에서 스스로 거부한다."""
    from pipeline.ingest.kis import KisSource
    from pipeline.ingest.price_sources import FdrSource
    root = DATA / "raw" / "prices"
    if name == "fdr":
        return FdrSource(root)
    if name == "kis":
        return KisSource(root / "kis")
    raise SystemExit(f"알 수 없는 가격 소스: {name} (fdr | kis)")


def cmd_probe_price(a) -> int:
    """소량으로 가격 소스를 검증한다. **대량 수집 전에 반드시 통과해야 한다.**

    추측으로 배선한 API 로 2,598종목을 돌리면 실패를 알아채는 시점이
    레이트 리밋을 소진한 뒤가 된다(DART 키 37자 절단을 프로브로 알았다).
    """
    from pipeline.ingest.price_sources import probe
    tickers = a.tickers or ["005930", "000850", "001130", "058430"]
    src = _price_source(a.source)
    r = probe(src, tickers, a.as_of)
    print(f"[{r.source}] {'통과' if r.ok else '실패'} — {r.detail}")
    if r.sample is not None:
        print(r.sample.to_string(index=False))
    if not r.ok:
        print("\n대량 수집은 거부된다. 위 오류를 먼저 해결할 것.", file=sys.stderr)
        return 1
    print("\n이 결과가 실제 시세와 맞는지 눈으로 확인한 뒤 "
          f"`ingest-prices --source {a.source}` 를 쓸 것.")
    return 0


def _available_sources() -> set[str]:
    """수집이 끝난 선택적 데이터 소스. requires_source 체크가 이걸 본다.

    목록을 코드에 두는 이유는 '수집했다고 설정에 적기만 하면 통과'되는
    구멍을 만들지 않기 위해서다. 실제 적재 여부를 스토어에서 확인한다.
    """
    have: set[str] = set()
    if not DB.exists():
        return have
    s = _store()
    try:
        n = s.con.execute(
            "SELECT count(*) FROM facts_financial WHERE statement_basis='DART_ALOT'"
        ).fetchone()[0]
        if n:
            have.add("dart_alot_matter")
        if s.con.execute(
            "SELECT count(*) FROM facts_financial "
            "WHERE element IN ('OWNER_STAKE_PCT','LARGEST_HOLDER_PCT')"
        ).fetchone()[0]:
            have.add("dart_hyslr_sttus")
    finally:
        s.close()
    return have


def _apply_check_overrides(specs, a):
    """--enable / --disable 로 설정을 일시 재정의. 없는 id 는 조용히 넘기지 않는다."""
    from dataclasses import replace
    known = {s.id for s in specs}
    for flag, ids in (("--enable", getattr(a, "enable", None) or []),
                      ("--disable", getattr(a, "disable", None) or [])):
        unknown = set(ids) - known
        if unknown:
            raise SystemExit(f"{flag}: 존재하지 않는 체크 {sorted(unknown)} "
                             f"(사용 가능: {sorted(known)})")
    on, off = set(getattr(a, "enable", None) or []), set(getattr(a, "disable", None) or [])
    return [replace(s, enabled=True) if s.id in on
            else replace(s, enabled=False) if s.id in off else s
            for s in specs]


def _screen_arg(parser) -> None:
    from pipeline.screen.registry import add_screen_arg
    add_screen_arg(parser)


def _paths(a):
    """스크린 인자를 경로 묶음으로. CLI 는 경로를 직접 조립하지 않는다."""
    from pipeline.screen.registry import resolve
    return resolve(getattr(a, "screen", None), REPO, DATA)


def cmd_checks(a) -> int:
    """사용 가능한 체크와 현재 on/off, 그리고 데이터 유무를 보여준다."""
    from pipeline.screen.checks import describe, load_checks
    p = _paths(a)
    cfg = yaml.safe_load(p.config.read_text(encoding="utf-8"))
    specs = load_checks(cfg.get("checks"))
    src = p.metrics(a.as_of) if a.as_of else None
    m = pd.read_parquet(src) if src and src.exists() else None
    if src and not src.exists():
        print(f"경고: {src} 없음 — 데이터 유무는 확인하지 못한다.", file=sys.stderr)
    print(f"[스크린 {p.screen}] {cfg.get('name') or ''}")
    print(describe(specs, m))
    print(f"\n설정: {p.config.relative_to(REPO)} 의 checks: 블록")
    print("일시 변경: pipeline screen --as-of <date> --enable <id> --disable <id>")
    print("임계값 튜닝: pipeline screen --as-of <date> --preview")
    return 0


def cmd_screen(a) -> int:
    p = _paths(a)
    cfg = yaml.safe_load(p.config.read_text(encoding="utf-8"))
    src = p.metrics(a.as_of)
    if not src.exists():
        print(f"{src} 없음 — 먼저 derive 를 실행할 것.", file=sys.stderr)
        return 1
    m = pd.read_parquet(src)

    from pipeline.screen.checks import (apply_checks, assert_gate_coverage,
                                        assert_monotone, enabled_gate_filters,
                                        enabled_hard_guards, load_checks, preview,
                                        risk_groups)
    specs = load_checks(cfg.get("checks"))
    specs = _apply_check_overrides(specs, a)
    if getattr(a, "preview", False):
        ok, _ = apply_hard_guards(m, cfg["universe"]["exclude_flags"],
                                  cfg["universe"]["min_market_cap_krw"])
        print(preview(m, specs, eligible=ok))
        return 0

    # 체크를 먼저 평가해 컬럼을 만든 뒤에 하드 가드를 건다 —
    # hard_guard 로 승격된 체크는 그 컬럼이 있어야 강제된다.
    m, check_rep = apply_checks(m, specs, available_sources=_available_sources())

    pending = cfg["universe"].get("pending_guards") or []
    if pending:
        print(f"경고: 미강제 가드 {pending} — 데이터 소스가 없어 적용되지 않는다.",
              file=sys.stderr)
    all_guards = list(cfg["universe"]["exclude_flags"]) + enabled_hard_guards(specs)
    guard_hits = guard_effectiveness(m, all_guards)
    eligible, why = apply_hard_guards(
        m, all_guards, cfg["universe"]["min_market_cap_krw"],
        include_sectors=cfg["universe"].get("include_sectors"))
    # 컬럼이 있어 '돌고는' 있지만 한 건도 배제하지 않는 가드를 드러낸다.
    # 지켜주고 있다고 믿는데 실은 아무것도 안 막는 상태가 가장 위험하다.
    inert = [g for g, n in guard_hits.items() if n == 0]
    if inert:
        print(f"주의: 배제 0건인 하드 가드 {inert} — 유니버스 구성상 무력할 수 있다. "
              f"제거하지 말고 왜 0인지 확인할 것.", file=sys.stderr)
    gate_cfg = dict(cfg["gate"])
    if a.target:
        gate_cfg["target_count"] = a.target
    assert_monotone(eligible, specs)
    assert_gate_coverage(eligible, specs)

    from pipeline.screen import basket as bk

    # 히스테리시스 기준은 '직전 실행'이 아니라 '직전 **리밸런스**'다.
    # 일별로 돌리면 기준이 매일 어제로 밀려 바스켓이 서서히 표류한다 —
    # 임계값 근처 종목의 잦은 교체를 막으려던 장치가 정반대로 작동한다.
    basket_path = p.basket
    legacy = basket_path.parent / "_last_members.json"
    current_basket = bk.load(basket_path if basket_path.exists() else legacy)

    res = run_gate(eligible, filters_from_config(cfg) + enabled_gate_filters(specs),
                   gate_cfg, cfg["ranking"]["weights"],
                   previous_members=current_basket.members)

    # 게재 축 — tier(검증)와 섞지 않는다. digest 가 본문 게재를 판정할 때 쓴다.
    res.survivors = res.survivors.assign(
        risk_groups=risk_groups(res.survivors, specs).values)

    d = p.screen_dir(a.as_of)
    d.mkdir(parents=True, exist_ok=True)
    res.survivors.to_parquet(d / "survivors.parquet", index=False)
    why.to_parquet(d / "why_excluded.parquet", index=False)
    # 해결된 임계값을 남기지 않으면 이 실행은 재현 불가다.
    manifest = {"as_of": str(a.as_of), "screen": p.screen,
                "config_version": cfg["version"], **res.manifest()}
    manifest["counts"]["universe"] = len(m)     # 다이제스트 퍼널의 첫 칸
    # 꺼진 체크도 기록한다. 결과만 보고 '무엇이 돌았지'를 되묻는 상황이 없어야 한다.
    manifest["checks"] = check_rep.manifest()
    manifest["guards"] = {"excluded_by": guard_hits,
                          "inert": inert,
                          "pending": list(pending)}
    manifest["risk_groups"] = {
        str(k): int(v) for k, v in
        res.survivors["risk_groups"].value_counts().sort_index().items()}
    now = set(res.survivors["ticker"])
    enter, exit_ = current_basket.drift(now)
    manifest["basket"] = {
        "mode": "rebalance" if a.rebalance else "scan",
        "reference": (str(current_basket.rebalanced_at)
                      if current_basket.exists else None),
        "members": len(current_basket.members),
        "entry_candidates": sorted(enter),
        "exit_candidates": sorted(exit_),
    }
    (d / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # **스캔은 바스켓을 건드리지 않는다.** 리밸런스에서만 확정한다 —
    # 매 실행 갱신하면 히스테리시스 기준이 매일 어제로 밀린다.
    if a.rebalance:
        bk.save(basket_path, a.as_of, now, cfg.get("version"))

    print(f"[{p.screen}] 유니버스 {len(m)} → 하드가드 통과 {len(eligible)} "
          f"→ 스크린 {len(res.survivors)}")
    print(bk.describe(current_basket, now, a.as_of, a.rebalance))
    if not a.rebalance and (enter or exit_):
        print(f"  진입후보 {sorted(enter)[:8]}{' …' if len(enter) > 8 else ''}")
        print(f"  이탈후보 {sorted(exit_)[:8]}{' …' if len(exit_) > 8 else ''}")
        print("  → 확정하려면 `screen --rebalance` (그 뒤 enrich/tag/verify/golden)")
    if check_rep.enabled or check_rep.disabled:
        print(f"체크 켜짐 {len(check_rep.enabled)} / 꺼짐 {len(check_rep.disabled)}"
              f" — 생존 종목 중 걸린 수: "
              f"{ {k: int(res.survivors[k].sum()) for k in check_rep.hit_counts if k in res.survivors} }")
    print(json.dumps(res.manifest(), ensure_ascii=False, indent=2))
    if not res.converged:
        print("경고: 목표 개수로 수렴하지 못했다. 필터 loose/tight 범위를 넓힐 것.",
              file=sys.stderr)
    return 0


def cmd_ingest_disclosures(a) -> int:
    """공시 목록 수집 — 촉매의 앵커. 본문은 받지 않는다(목록만 100건/1콜)."""
    from pipeline.ingest.dart_disclosure import (CATALYST_TYPES,
                                                 DisclosureIngestor, chunk_ranges)
    end = a.end or date.today()
    bgn = a.begin or (end - timedelta(days=a.lookback))
    if bgn > end:
        print(f"기간이 거꾸로다: {bgn} > {end}", file=sys.stderr)
        return 1
    ing = DisclosureIngestor(DartClient(RAW_DART))
    s = _store()
    total_new = total_seen = 0
    try:
        for b, e in chunk_ranges(bgn, end, days=a.chunk_days):
            df = ing.fetch_range(b, e, types=tuple(a.types or CATALYST_TYPES),
                                 refresh=a.refresh)
            total_seen += len(df)
            total_new += s.upsert_disclosures(df)
            print(f"  {b}~{e}  수집 {len(df):>5}건")
        n = s.con.execute("SELECT count(*) FROM disclosures").fetchone()[0]
        by = s.con.execute("""SELECT pblntf_ty, count(*) n FROM disclosures
                              GROUP BY 1 ORDER BY 1""").df()
    finally:
        s.close()
    print(f"\n조회 {total_seen}건 | 신규 {total_new}건 | 누적 {n}건")
    print(by.to_string(index=False))
    print(f"DART 잔여 호출 {ing.client.remaining_calls():,}")
    return 0


def cmd_status(_a) -> int:
    if not DB.exists():
        print("PIT 스토어 없음. init 을 먼저 실행할 것.")
        return 1
    s = _store()
    for t in ("facts_financial", "prices", "security_master", "status_events"):
        n = s.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t:20} {n:>10,}")
    r = s.con.execute("""
        SELECT min(reported_at), max(reported_at), count(DISTINCT ticker)
        FROM facts_financial
    """).fetchone()
    if r[0]:
        print(f"공시 접수일 범위      {r[0]} ~ {r[1]}  (종목 {r[2]:,})")
    rev = s.con.execute(
        "SELECT count(*) FROM facts_financial WHERE revision_of IS NOT NULL").fetchone()[0]
    print(f"정정공시 fact         {rev:>10,}")
    s.close()
    return 0


# ── M2: 엔리치먼트 → 태깅 → 검증 → 다이제스트 ──────────────────────
def cmd_enrich(a) -> int:
    """생존 종목의 evidence pack 생성. 사업보고서 원문은 요약하지 않고 섹션 발췌한다."""
    from pipeline.enrich.evidence import build_pack
    from pipeline.ingest.dart_document import (MIN_PLAUSIBLE_SECTION as MIN_SEC,
                                                DocumentClient, extract_business_section)

    p = _paths(a)
    src = p.screen_dir(a.as_of) / "survivors.parquet"
    if not src.exists():
        print(f"{src} 없음 — 먼저 screen --screen {p.screen} 을 실행할 것.",
              file=sys.stderr)
        return 1
    surv = pd.read_parquet(src)
    if a.limit:
        surv = surv.head(a.limit)

    st = _store()
    # 사업보고서 접수번호: FY 재무 fact 의 source_doc_id 를 재사용한다(추가 조회 불필요)
    rc = st.con.execute("""
        SELECT ticker, any_value(source_doc_id) AS rcept_no FROM (
            SELECT ticker, source_doc_id, row_number() OVER (
                PARTITION BY ticker ORDER BY fiscal_end_date DESC, reported_at DESC) rn
            FROM facts_financial WHERE period_type = 'FY')
        WHERE rn = 1 GROUP BY ticker
    """).df()
    st.close()
    rmap = dict(zip(rc["ticker"], rc["rcept_no"]))

    doc = DocumentClient(DATA / "raw" / "dart_doc")
    out_root = p.enrich_dir(a.as_of)
    ok, no_section, failed, short = 0, 0, 0, 0
    for row in surv.itertuples():
        rcept = rmap.get(row.ticker)
        if not rcept:
            failed += 1
            continue
        try:
            markup = doc.fetch(rcept, refresh=a.refresh)
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  {row.ticker} 원문 실패: {str(e)[:90]}", file=sys.stderr)
            continue
        ex = extract_business_section(markup, max_chars=a.max_chars)
        full = extract_business_section(markup, max_chars=10 ** 9)   # 세그먼트 추출용
        if not ex.found_section:
            no_section += 1
        if ex.suspiciously_short:
            short += 1
        meta = {"rcept_no": rcept, "found_section": ex.found_section,
                "suspiciously_short": ex.suspiciously_short,
                "truncated_chars": ex.truncated_chars, "total_chars": ex.total_chars,
                "source_url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept}"}
        pack = build_pack(pd.Series(row._asdict()), ex.text, meta, [], a.as_of,
                          segment_source=full.text)
        pack.write(out_root)
        ok += 1

    print(f"evidence pack {ok}개 → {out_root}")
    print(f"  섹션 미발견 {no_section} (본문 앞부분으로 대체, meta.found_section=false)")
    print(f"  본문 과소({MIN_SEC}자 미만) {short}건 - 목차를 잡았을 가능성, LLM 입력 부적합")
    print(f"  실패 {failed}")
    return 0


def _load_packs(paths, as_of) -> list[dict]:
    import json as _json
    root = paths.enrich_dir(as_of)
    out = []
    for f in sorted(root.glob("*/pack.json")):
        out.append(_json.loads(f.read_text(encoding="utf-8")))
    return out


def cmd_tag(a) -> int:
    """캐스케이드 태깅. Haiku 1차 → 조건부 Sonnet 승급."""
    import asyncio, json as _json
    from pipeline.enrich.evidence import EvidencePack
    from pipeline.llm.cascade import TagCache, tag_universe

    import os
    if not a.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY 가 없다. 리포 루트 .env 에 다음 줄을 추가할 것:\n"
              "  ANTHROPIC_API_KEY=sk-ant-...\n"
              "(.env 는 .gitignore 대상이다. 비용 없이 형태만 보려면 --dry-run)",
              file=sys.stderr)
        return 1

    p = _paths(a)
    packs_raw = _load_packs(p, a.as_of)
    if not packs_raw:
        print(f"evidence pack 없음 — 먼저 enrich 를 실행할 것.", file=sys.stderr)
        return 1
    if a.limit:
        packs_raw = packs_raw[:a.limit]

    taxonomy = yaml.safe_load(
        (REPO / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))
    sectors = yaml.safe_load(
        (REPO / "configs/sectors/sector_map.yaml").read_text(encoding="utf-8"))
    universe = set(sectors["codes"])

    packs = [EvidencePack(**p).to_cascade_input() for p in packs_raw]

    if a.dry_run:
        return _tag_dry_run(packs, taxonomy, universe)

    # 캐시는 **스크린 간 공유**다. pack_hash 키라 같은 종목이 두 스크린에
    # 걸리면 두 번째는 API 호출 0회로 끝난다.
    cache = TagCache(p.tag_cache)
    results = asyncio.run(tag_universe(packs, taxonomy, universe, cache))

    out = p.tags(a.as_of)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    esc = sum(1 for r in results if r.get("escalated_for"))
    ab = sum(1 for r in results if r.get("abstain"))
    n_asg = sum(len(r.get("assignments") or []) for r in results)
    print(f"태깅 {len(results)}종목 → {out}")
    print(f"  배정 {n_asg}건 | 승급 {esc} ({esc/max(len(results),1):.0%}) | 기권 {ab}")
    return 0


# Haiku 4.5 / Sonnet 5 입력·출력 단가 ($/MTok). 캐시 읽기는 입력가의 약 0.1배.
PRICE = {"claude-haiku-4-5": (1.0, 5.0), "claude-sonnet-5": (3.0, 15.0)}
ESCALATION_ASSUMED = 0.30
OUTPUT_TOKENS_ASSUMED = 400


def _tag_dry_run(packs, taxonomy, universe) -> int:
    """LLM 호출 없이 Stage 0 결과와 비용 추정만 낸다. Anthropic 키 불필요."""
    from collections import Counter
    from pipeline.llm.cascade import (MODEL_CHEAP, MODEL_DEEP, build_system,
                                      build_user, narrow_candidates)

    prefix_chars = sum(len(b["text"]) for b in build_system(taxonomy))
    prefix_tok = int(prefix_chars / 2.2)

    themes, no_cand, per_stock_tok = Counter(), 0, []
    for pk in packs:
        cands = narrow_candidates(pk, taxonomy, universe)
        if not cands:
            no_cand += 1
            continue
        themes.update(c.theme_id for c in cands)
        per_stock_tok.append(int(len(build_user(pk, cands)) / 2.2))

    n = len(packs)
    body = sum(per_stock_tok)
    avg = body // max(len(per_stock_tok), 1)

    def cost(model, n_calls, cached):
        pin, pout = PRICE[model]
        # 프리픽스는 첫 호출만 쓰기(1.25x), 나머지는 캐시 읽기(0.1x)
        pre = prefix_tok * (1.25 + 0.1 * (n_calls - 1)) if cached else prefix_tok * n_calls
        return ((pre + body * n_calls / max(n, 1)) / 1e6 * pin
                + n_calls * OUTPUT_TOKENS_ASSUMED / 1e6 * pout)

    esc = int(n * ESCALATION_ASSUMED)
    c1, c2 = cost(MODEL_CHEAP, n, True), cost(MODEL_DEEP, esc, True)

    print("[dry-run] LLM 호출 없음\n")
    print(f"대상 {n}종목 | 후보 테마 없음 {no_cand}")
    print(f"캐시 프리픽스(규칙+사전) ≈ {prefix_tok:,} 토큰")
    print(f"종목당 본문 ≈ {avg:,} 토큰 (합 {body:,})")
    print("\nStage0 후보 테마 상위 12:")
    for t, c in themes.most_common(12):
        print(f"  {c:>4}  {t}")
    print(f"\n비용 추정 (승급률 {ESCALATION_ASSUMED:.0%} 가정, 프롬프트 캐시 적용)")
    print(f"  Stage1 {MODEL_CHEAP:<20} {n:>4}콜  ${c1:,.2f}")
    print(f"  Stage2 {MODEL_DEEP:<20} {esc:>4}콜  ${c2:,.2f}")
    print(f"  합계                            ${c1 + c2:,.2f}")
    print(f"  Batch API 적용 시 (토큰 50% 할인)  ${(c1 + c2) * 0.5:,.2f}")
    print("\n※ 토큰은 문자수 기반 추정. ANTHROPIC_API_KEY 설정 후 실측 권장.")
    return 0


def cmd_verify(a) -> int:
    import json as _json
    from dataclasses import asdict
    from pipeline.verify.layers import SectorMatrix, verification_metrics, verify_batch

    p = _paths(a)
    tags_p = p.tags(a.as_of)
    if not tags_p.exists():
        print(f"{tags_p} 없음 — 먼저 tag 를 실행할 것.", file=sys.stderr)
        return 1
    tags = _json.loads(tags_p.read_text(encoding="utf-8"))
    packs = {k["ticker"]: k for k in _load_packs(p, a.as_of)}
    surv = pd.read_parquet(p.screen_dir(a.as_of) / "survivors.parquet")
    sectors = dict(zip(surv["ticker"], surv["sector_code"]))

    verdicts = verify_batch(tags, packs, sectors, SectorMatrix())
    df = pd.DataFrame([asdict(v) for v in verdicts])
    d = p.verify_dir(a.as_of)
    d.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df = df.assign(flags=df["flags"].map(lambda x: "|".join(x)),
                       segments=df["segments"].map(lambda x: "|".join(x)))
        df.to_parquet(
            d / "verdicts.parquet", index=False)
    metrics = verification_metrics(verdicts, n_stocks=len(packs))
    (d / "metrics.json").write_text(
        _json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(_json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


# 골든셋 라벨과 측정 대상의 최소 겹침. 이보다 낮으면 다른 코호트를 재는 것이다.
MIN_COHORT_OVERLAP = 0.80


def cmd_golden(a) -> int:
    """골든셋 대비 오분류율. 핵심은 절대값이 아니라 검증 전후 차이다."""
    import json as _json
    from pipeline.verify.golden import (cores_from_tags, cores_from_verdicts,
                                        load_golden, score)

    p = _paths(a)
    # 라벨 파일은 스크린마다 다르다. 없는 스크린에서 딥밸류 라벨을 쓰면
    # 겹침 0% 를 '코호트 표류'로 오진하게 된다 — 실제로는 라벨이 없는 것이다.
    label_paths = [Path(a.golden)] if a.golden else p.golden
    if not label_paths:
        print(f"스크린 '{p.screen}' 의 골든셋 라벨이 없다.\n"
              f"  오분류율은 티커 단위 라벨 없이는 잴 수 없다. 다음 중 하나를 할 것:\n"
              f"    · tests/golden/ 에 이 스크린 코호트용 라벨을 만들고 "
              f"registry.GOLDEN_LABELS 에 등록\n"
              f"    · --golden <경로> 로 직접 지정", file=sys.stderr)
        return 1
    # 여러 파일을 합친다. **같은 티커가 두 곳에 있으면 실패시킨다** — 조용히
    # 덮어쓰면 어느 쪽이 정답인지 모르는 채로 지표가 나오고, 라벨을 고쳐도
    # 반영되지 않는 파일이 생긴다.
    gold: dict = {}
    for lp in label_paths:
        part = load_golden(lp)
        if dup := set(gold) & set(part):
            print(f"라벨 중복: {sorted(dup)[:5]} 가 여러 파일에 있다 "
                  f"({lp.name} 포함). 티커당 한 곳에만 둘 것.", file=sys.stderr)
            return 1
        gold.update(part)
    label_path = label_paths[0]
    tags_p = p.tags(a.as_of)
    if not tags_p.exists():
        print(f"{tags_p} 없음 — 먼저 tag 를 실행할 것.", file=sys.stderr)
        return 1
    tags = _json.loads(tags_p.read_text(encoding="utf-8"))
    vp = p.verify_dir(a.as_of) / "verdicts.parquet"
    if not vp.exists():
        print(f"{vp} 없음 — 먼저 verify 를 실행할 것.", file=sys.stderr)
        return 1
    verdicts = pd.read_parquet(vp).to_dict("records")

    # ── 코호트 정합성 ────────────────────────────────────────────
    # 골든셋 라벨은 **티커 단위**다. 리밸런스로 바스켓이 바뀐 뒤 측정하면
    # 오분류율이 '데이터 품질'이 아니라 '오늘의 종목 구성'을 재게 된다.
    # 겹침이 낮은데 조용히 숫자를 내놓으면 그 숫자를 신뢰하게 되므로 막는다.
    labeled = set(gold)
    tagged = {t["ticker"] for t in tags}
    overlap = labeled & tagged
    # **분모는 측정 대상이다.** 위험한 방향은 '태깅됐는데 라벨이 없는' 쪽이다 —
    # 그 종목들은 분모 밖으로 조용히 빠져 지표가 코호트의 일부만 재게 된다.
    # 반대 방향(라벨은 있는데 이번에 안 걸린 종목)은 측정의 타당성을 해치지
    # 않는다. GARP·Quality FCF 처럼 두 스크린이 라벨 파일을 공유하면 항상
    # 발생하므로, 이걸 분모로 삼으면 정상 상태에서 매번 막힌다.
    cover = len(overlap) / len(tagged) if tagged else 0.0
    print(f"코호트: 라벨 {len(labeled)} / 측정 대상 {len(tagged)} / "
          f"겹침 {len(overlap)} (측정 대상의 {cover:.0%})")
    if missing := labeled - tagged:
        print(f"  라벨은 있으나 이번 코호트에 없음 {len(missing)}건: "
              f"{sorted(missing)[:5]} — 타당성에는 영향 없음", file=sys.stderr)
    if extra := tagged - labeled:
        print(f"  태깅됐으나 라벨이 없음 {len(extra)}건: {sorted(extra)[:5]} "
              f"— 이 종목들은 지표에 반영되지 않는다(분모 밖)", file=sys.stderr)
    if cover < MIN_COHORT_OVERLAP and not a.allow_cohort_drift:
        lp = label_path.name
        print(f"\n중단: 측정 대상 {len(tagged)}종목 중 {cover:.0%} 만 라벨이 있다 "
              f"(기준 {MIN_COHORT_OVERLAP:.0%}). 라벨 없는 종목은 분모 밖이라 "
              f"지금 내는 숫자는 코호트의 일부만 잰다.\n"
              f"  tests/golden/{lp} 에 {len(extra)}종목을 추가하거나 "
              f"--allow-cohort-drift 로 감수할 것.", file=sys.stderr)
        return 1

    from pipeline.verify.golden import REPRODUCIBILITY as RP

    # **채점은 이번 코호트로 한정한다.** score() 는 gold 를 전부 훑으며 예측이
    # 없는 라벨을 FN 으로 센다. 여러 파일을 합치면 다른 스크린의 라벨까지
    # 분모에 들어가 재현율이 무너진다(실측: 80.3% → 27.2%). 그건 분류 성능이
    # 아니라 '이 스크린이 남의 코호트를 안 뽑았다'는 사실을 재는 것이다.
    gold = {k: v for k, v in gold.items() if k in tagged}

    defects = [g for g in gold.values() if g.evidence_defect]
    contested = [g for g in gold.values() if g.confidence == "contested"]

    # ── 재현 가능한 행 ───────────────────────────────────────────
    # LLM 의 core 선택은 **이산 선택**이라 재실행해도 같다(실측 24/24).
    # 이 값만 개선 지표로 쓸 수 있다 — 두 실행의 차이가 곧 변경의 효과다.
    primary = [
        score(cores_from_tags(tags), gold, "검증 전 (LLM 원본 core)",
              reproducible=True),
        score(cores_from_tags(tags), gold, "  (certain 라벨만)",
              certain_only=True, reproducible=True),
    ]
    # ── 재현되지 않는 행 ─────────────────────────────────────────
    # 전부 cores_from_verdicts 를 거친다. V3 가 LLM 이 뱉은 revenue_share_claim
    # (연속값)을 임계와 대조하므로, 그 값이 임계 근처에서 흔들리면 하드 플래그가
    # 켜졌다 꺼지고 등급이 뒤집힌다(실측 tier 일치 84%).
    secondary = [
        score(cores_from_verdicts(verdicts), gold, "검증 후 (폐기·강등 반영)"),
        # 오탐 분모가 한 자릿수라 라벨 1건이 지표를 뒤집는다. 판정이 갈릴 수 있는
        # 라벨을 뺀 값을 함께 보여, 숫자가 라벨 판단에 얼마나 기대는지 드러낸다.
        score(cores_from_verdicts(verdicts), gold,
              "  (certain 라벨만)", certain_only=True),
        # 비중 근거로 좁힌 값. 재현율이 낮은 것은 분류 실패가 아니라
        # **세그먼트를 공시하지 않는 회사가 많다**는 사실이다.
        score(cores_from_verdicts(verdicts, share_evidence={"confirmed"}), gold,
              "  비중 실측 확인분만"),
        # 플래그에 신호가 있는지 보는 대조군.
        score(cores_from_verdicts(verdicts, clean_only=True), gold,
              "  (대조) 플래그 0건만"),
        score(cores_from_verdicts(verdicts), gold,
              "검증 후 (evidence 결함 제외)", skip_evidence_defect=True),
    ]
    runs = primary + secondary

    def fmt(x):
        return "  n/a" if x is None else f"{x:6.1%}"

    def table(rows):
        for s in rows:
            print(f"{s.label:<28} {fmt(s.precision)} {fmt(s.recall)} "
                  f"{fmt(s.misclassification_rate)} "
                  f"{s.tp:>4} {s.fp:>4} {s.fn:>4}  {s.correct_abstain}")

    head = (f"{'단계':<28} {'정밀도':>8} {'재현율':>8} {'오분류율':>9} "
            f"{'TP':>4} {'FP':>4} {'FN':>4}  기권정확")
    print(f"\n골든셋: {len(gold)}종목 "
          f"(core 보유 {sum(1 for g in gold.values() if g.core)}, "
          f"core 없음 {sum(1 for g in gold.values() if not g.core)}, "
          f"evidence 결함 {len(defects)})")

    print(f"\n■ 기준 지표 — core 일치 {RP['core_agreement']:.0%} "
          f"(누적 n={RP['n_stocks']}, 표본 {RP['samples']}회). "
          f"재현율 ±{RP['recall_noise_pp']}%p 보다 작은 차이는 잡음이다")
    print(head)
    print("-" * 88)
    table(primary)

    print(f"\n■ 참고 — 검증 반영. 판정 일치 {RP['verdict_agreement']:.0%} "
          f"(누적 배정 {RP['n_assignments']}건) — 기준 지표보다 더 흔들린다")
    print(head)
    print("-" * 88)
    table(secondary)
    print(f"  ※ 등급(A/B/C)과 share_overclaim 게이트를 없애 흔들림의 두 원인은 "
          f"제거했으나, 역할 강등이 세그먼트 이름 매칭(LLM 출력)에\n"
          f"     기대는 부분이 남아 있다. 승급 호출(85%)은 온도 고정이 불가하다. "
          f"재측정: {RP['tool']}")

    after = secondary[0]
    print("\n오탐 상위:", after.as_dict()["top_false_positives"] or "없음")
    print("미탐 상위:", after.as_dict()["top_false_negatives"] or "없음")
    if contested:
        print(f"\n판정이 갈릴 수 있는 라벨 {len(contested)}건 "
              f"— certain-only 행과 대조해 지표가 이들에 얼마나 기대는지 볼 것:")
        for g in contested:
            why = g.note.split("갈리는 이유:")[-1].strip()
            print(f"  {g.ticker} {g.name:<12} {sorted(g.core)}")
            print(f"       {why[:88]}")
    if defects:
        print("\nevidence 결함(LLM 이 아니라 입력이 틀린 케이스):")
        for g in defects:
            print(f"  {g.ticker} {g.name} — {g.note}")

    out = p.verify_dir(a.as_of) / "golden_metrics.json"
    out.write_text(_json.dumps(
        {"golden_stocks": len(gold), "evidence_defects": len(defects),
         "contested": [g.ticker for g in contested],
         # 소비자가 행마다 reproducible 을 보고 판단할 수 있어야 한다.
         # 실측값을 같이 실어 나중에 이 파일만 봐도 해석이 가능하게 한다.
         "reproducibility": RP,
         "runs": [s.as_dict() for s in runs]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")
    return 0


def cmd_catalysts(a) -> int:
    """공시 → 촉매. LLM 호출 0회.

    magnitude 를 못 구하면 촉매를 만들지 않는다 — 0 으로 채우면 '작은 촉매'와
    '크기를 모르는 촉매'가 같아진다(세그먼트 미공시를 0% 로 읽지 않는 것과 같다).
    """
    from pipeline.catalysts.build import (amendment_index, build, classify,
                                          load_catalog)
    from pipeline.catalysts.extract import (STRUCTURED, from_contract_document,
                                            from_structured)
    from pipeline.ingest.dart_document import DocumentClient, to_plain_text

    p = _paths(a)
    src = p.metrics(a.as_of)
    if not src.exists():
        print(f"{src} 없음 — 먼저 derive 를 실행할 것.", file=sys.stderr)
        return 1
    metrics = pd.read_parquet(src)

    scope = None
    if not a.all_universe:
        sp = p.screen_dir(a.as_of) / "survivors.parquet"
        if not sp.exists():
            print(f"{sp} 없음 — screen 을 먼저 돌리거나 --all-universe 를 쓸 것.",
                  file=sys.stderr)
            return 1
        scope = set(pd.read_parquet(sp)["ticker"])

    specs_list, _cfg = load_catalog()
    specs = {s.id: s for s in specs_list}
    since = a.as_of - timedelta(days=a.lookback)

    st = _store()
    try:
        disc = st.disclosures_asof(a.as_of, since=since)
    finally:
        st.close()
    if disc.empty:
        print("공시가 없다 — ingest-disclosures 를 먼저 실행할 것.", file=sys.stderr)
        return 1

    # 정정 인덱스는 **정정본을 포함한 전체**에서 만든다. 원본만 보면
    # '이 건이 나중에 뒤집혔나' 를 알 수 없다.
    amended = amendment_index(disc)
    cands = classify(disc[~disc["is_amendment"]], specs_list)
    if scope is not None:
        cands = cands[cands["ticker"].isin(scope)]
    print(f"공시 {len(disc):,}건({since}~{a.as_of}) → 촉매 후보 {len(cands):,}건 "
          f"/ {cands['ticker'].nunique() if len(cands) else 0}종목")
    if cands.empty:
        return 0

    close = metrics.set_index("ticker")["close"].to_dict()
    mags: dict = {}

    cli = DartClient(RAW_DART)
    for kind in sorted(set(cands["kind"]) & set(STRUCTURED)):
        sub = cands[cands["kind"] == kind]
        for corp, g in sub.groupby("corp_code"):
            got = from_structured(cli, kind, corp, since, a.as_of,
                                  close=close.get(g["ticker"].iloc[0]))
            for rc, mg in got.items():
                mags[rc] = (mg.value, mg.basis, mg.expires_at)
        hit = sum(1 for r in sub["rcept_no"] if r in mags)
        print(f"  {kind} 구조화 API: {len(sub)}건 중 금액 확보 {hit}건")

    c2 = cands[cands["kind"] == "C2"]
    if not c2.empty:
        doc = DocumentClient(DATA / "raw" / "dart_doc")
        ok = 0
        for rc in c2["rcept_no"]:
            try:
                mg = from_contract_document(to_plain_text(doc.fetch(rc)))
            except Exception:                                # noqa: BLE001
                continue
            if mg:
                mags[rc] = (mg.value, mg.basis, mg.expires_at)
                ok += 1
        print(f"  C2 문서 파싱: {len(c2)}건 중 {ok}건 성공")

    cats = build(cands, specs, mags, metrics, a.as_of, amended)
    keep = [c for c in cats
            if specs[c.kind].mag_min is None or c.magnitude is not None]
    dropped = len(cats) - len(keep)

    df = pd.DataFrame([c.to_row() for c in keep])
    d = p.screen_dir(a.as_of)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "catalysts.parquet", index=False)

    print(f"\n촉매 {len(df):,}건 (크기 미상으로 제외 {dropped}건) "
          f"→ {d / 'catalysts.parquet'}")
    if not df.empty:
        print(df.groupby(["kind", "name"]).agg(
            건수=("ticker", "size"), 종목=("ticker", "nunique"),
            신뢰도평균=("confidence", "mean")).round(2).to_string())
        print(f"\n신뢰도 분포: {df['confidence'].value_counts().sort_index().to_dict()}")
    return 0


def cmd_report(a) -> int:
    import json as _json
    from pipeline.report.digest import render

    p = _paths(a)
    d = p.screen_dir(a.as_of)
    surv = pd.read_parquet(d / "survivors.parquet")
    funnel = _json.loads((d / "manifest.json").read_text(encoding="utf-8")).get("counts", {})
    vd = p.verify_dir(a.as_of)
    verdicts = (pd.read_parquet(vd / "verdicts.parquet")
                if (vd / "verdicts.parquet").exists() else pd.DataFrame())
    vmetrics = (_json.loads((vd / "metrics.json").read_text(encoding="utf-8"))
                if (vd / "metrics.json").exists() else {})
    qp = DATA / "derived" / f"quality_{a.as_of}.json"   # 파생은 스크린 무관
    quality = _json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else {}
    taxonomy = yaml.safe_load(
        (REPO / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))

    screen_manifest = _json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    screen_cfg = yaml.safe_load(p.config.read_text(encoding="utf-8"))
    # 촉매는 선택이다 — `catalysts` 를 안 돌린 스크린도 다이제스트는 나와야 한다.
    cp = d / "catalysts.parquet"
    cat = pd.read_parquet(cp) if cp.exists() else None
    md = render(a.as_of, surv, verdicts, taxonomy, funnel, quality, vmetrics,
                checks=screen_manifest.get("checks"),
                max_risk_groups=(screen_cfg.get("digest") or {})
                .get("max_risk_groups_for_body", 2),
                screen_name=f"{screen_cfg.get('name') or p.screen} 스크린",
                catalysts=cat)
    out = p.out_dir(a.as_of)
    out.mkdir(parents=True, exist_ok=True)
    (out / "digest.md").write_text(md, encoding="utf-8")
    print(f"다이제스트 → {out / 'digest.md'}  ({len(md):,}자)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)
    im = sub.add_parser("ingest-master")
    im.add_argument("--refresh", action="store_true", help="raw 캐시 무시하고 재수집")
    im.add_argument("--with-dart", action="store_true", help="DART corp_code 도 매칭(키 필요)")
    im.set_defaults(fn=cmd_ingest_master)
    idc = sub.add_parser("ingest-disclosures",
                         help="공시 목록 수집 (촉매 앵커). 목록만, 본문 아님")
    idc.add_argument("--lookback", type=int, default=180,
                     help="오늘부터 며칠 전까지 (기본 180). --begin 이 있으면 무시")
    idc.add_argument("--begin", type=_iso)
    idc.add_argument("--end", type=_iso)
    idc.add_argument("--types", nargs="*",
                     help="공시유형 (기본 B I). A=정기 B=주요사항 I=거래소")
    idc.add_argument("--chunk-days", type=int, default=14,
                     help="한 요청의 기간. 길면 페이지가 폭주한다")
    idc.add_argument("--refresh", action="store_true")
    idc.set_defaults(fn=cmd_ingest_disclosures)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    rn = sub.add_parser("renormalize", help="캐시된 raw 로 facts 재생성 (API 호출 0회)")
    rn.add_argument("--lag", type=int, default=1)
    rn.set_defaults(fn=cmd_renormalize)

    d = sub.add_parser("ingest-dart")
    d.add_argument("--year", type=int, required=True)
    d.add_argument("--quarter", type=int, required=True, choices=[1, 2, 3, 4])
    d.add_argument("--limit", type=int, help="목록 앞에서 N개 (=최근 상장사)")
    d.add_argument("--sample", type=int, help="무작위 N개 표본")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--listed-before", help="이 날짜 이전 상장분만 (YYYY-MM-DD)")
    d.add_argument("--tickers", nargs="*")
    d.add_argument("--lag", type=int, default=1, help="available_at = reported_at + lag일")
    d.set_defaults(fn=cmd_ingest_dart)

    pr = sub.add_parser("ingest-prices")
    pr.add_argument("--as-of", type=_iso, required=True)
    pr.add_argument("--csv")
    pr.add_argument("--limit", type=int)
    pr.add_argument("--tickers", nargs="*")
    pr.add_argument("--source", default="fdr", choices=["fdr", "kis"],
                    help="주 가격 소스")
    pr.add_argument("--cross-check", nargs="*", metavar="SOURCE",
                    help="종가를 대조할 추가 소스. 없으면 주가 오류를 잡을 수 없다 "
                         "— 시총·PER·PBR 이 전부 같은 종가에서 나오기 때문")
    pr.add_argument("--with-facts", action="store_true",
                    help="재무가 적재된 종목만 대상으로")
    pr.add_argument("--workers", type=int, default=8)
    pr.add_argument("--refresh", action="store_true")
    pr.add_argument("--shares-year", type=int, default=2025)
    pr.add_argument("--shares-quarter", type=int, default=4, choices=[1, 2, 3, 4])
    pr.set_defaults(fn=cmd_ingest_prices)

    dv = sub.add_parser("derive")
    dv.add_argument("--as-of", type=_iso, required=True)
    dv.set_defaults(fn=cmd_derive)

    sc = sub.add_parser("screen")
    sc.add_argument("--as-of", type=_iso, required=True)
    sc.add_argument("--target", type=int)
    sc.add_argument("--enable", action="append", metavar="CHECK_ID",
                    help="체크를 이번 실행에만 켠다 (설정 파일은 그대로)")
    sc.add_argument("--disable", action="append", metavar="CHECK_ID")
    sc.add_argument("--rebalance", action="store_true",
                    help="바스켓을 이 시점으로 확정한다. 히스테리시스 기준이 갱신되고 "
                         "골든셋 측정 코호트가 바뀐다. 생략하면 스캔(바스켓 불변)")
    sc.add_argument("--preview", action="store_true",
                    help="스크린하지 않고 체크별 적중 수와 분포만 출력 (임계값 튜닝용)")
    _screen_arg(sc)
    sc.set_defaults(fn=cmd_screen)

    idv = sub.add_parser("ingest-dividend", help="DART 배당에 관한 사항 (종목당 1콜)")
    idv.add_argument("--year", type=int, required=True, help="사업연도")
    idv.add_argument("--lag", type=int, default=1)
    idv.add_argument("--limit", type=int)
    idv.add_argument("--tickers", nargs="*")
    idv.add_argument("--refresh", action="store_true", help="raw 캐시 무시하고 재수집")
    idv.set_defaults(fn=cmd_ingest_dividend)

    ihl = sub.add_parser("ingest-holder", help="DART 최대주주 현황 (종목당 1콜)")
    ihl.add_argument("--year", type=int, required=True)
    ihl.add_argument("--lag", type=int, default=1)
    ihl.add_argument("--limit", type=int)
    ihl.add_argument("--tickers", nargs="*")
    ihl.add_argument("--refresh", action="store_true")
    ihl.set_defaults(fn=cmd_ingest_holder)

    pp = sub.add_parser("probe-price", help="가격 소스 소량 검증 (대량 수집 전 필수)")
    pp.add_argument("--source", default="fdr", choices=["fdr", "kis"])
    pp.add_argument("--as-of", type=_iso, required=True)
    pp.add_argument("--tickers", nargs="*")
    pp.set_defaults(fn=cmd_probe_price)

    ck = sub.add_parser("checks", help="스크린 체크 목록·on/off·데이터 유무")
    ck.add_argument("--as-of", type=_iso, help="지정 시 데이터 유무까지 확인")
    _screen_arg(ck)
    ck.set_defaults(fn=cmd_checks)

    en = sub.add_parser("enrich")
    en.add_argument("--as-of", type=_iso, required=True)
    en.add_argument("--limit", type=int)
    en.add_argument("--max-chars", type=int, default=12000)
    en.add_argument("--refresh", action="store_true")
    _screen_arg(en)
    en.set_defaults(fn=cmd_enrich)

    tg = sub.add_parser("tag")
    tg.add_argument("--as-of", type=_iso, required=True)
    tg.add_argument("--limit", type=int)
    tg.add_argument("--dry-run", action="store_true",
                    help="LLM 호출 없이 Stage0 결과와 비용 추정만")
    _screen_arg(tg)
    tg.set_defaults(fn=cmd_tag)

    vf = sub.add_parser("verify")
    vf.add_argument("--as-of", type=_iso, required=True)
    _screen_arg(vf)
    vf.set_defaults(fn=cmd_verify)

    gd = sub.add_parser("golden", help="골든셋 대비 오분류율 (검증 전후 비교)")
    gd.add_argument("--as-of", type=_iso, required=True)
    gd.add_argument("--allow-cohort-drift", action="store_true",
                    help="라벨과 측정 대상이 어긋나도 진행한다(값의 의미가 달라짐)")
    gd.add_argument("--golden", help="라벨 jsonl 경로 (기본: tests/golden/kr_core_themes_v1.jsonl)")
    _screen_arg(gd)
    gd.set_defaults(fn=cmd_golden)

    ct = sub.add_parser("catalysts", help="공시 → 촉매 (LLM 호출 0회)")
    ct.add_argument("--as-of", type=_iso, required=True)
    ct.add_argument("--lookback", type=int, default=180)
    ct.add_argument("--all-universe", action="store_true",
                    help="스크린 통과 종목이 아니라 전 종목 대상")
    _screen_arg(ct)
    ct.set_defaults(fn=cmd_catalysts)

    rp = sub.add_parser("report")
    rp.add_argument("--as-of", type=_iso, required=True)
    _screen_arg(rp)
    rp.set_defaults(fn=cmd_report)

    a = p.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
