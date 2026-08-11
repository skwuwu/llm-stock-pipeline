#!/usr/bin/env python
"""매일 장 마감 후 시세를 갱신한다.

    python scripts/daily_prices.py                 # 마지막 거래일 자동 탐지
    python scripts/daily_prices.py --derive        # 파생지표까지
    python scripts/daily_prices.py --as-of 2026-08-07 --force

설계 메모
─────────
**as_of 는 오늘이 아니라 '마지막 거래일'이다.** build_price_table 이
`df["date"] = as_of` 로 저장하므로, 토요일에 오늘 날짜로 돌리면 금요일 종가가
토요일 행으로 들어간다. 휴장일에 유령 행이 생기고 수익률 계산이 깨진다.

거래일은 **공휴일 달력 대신 프로브로 정한다.** 달력은 매년 갱신해야 하고
임시휴장(2025-12-31 종가 없음 등)을 못 잡는다. 유동성이 확실한 종목 하나를
조회해 데이터가 실제로 존재하는 마지막 날짜를 쓰면 그런 유지보수가 없어진다.

수집이 부분적으로 실패해도 조용히 넘기지 않는다. 커버리지가 임계 미만이면
**비정상 종료**한다 — 스케줄러가 실패를 알아야 다음 날 이상을 눈치챈다.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
LOG = REPO / "data" / "logs" / "daily_prices.jsonl"

# 거래일 탐지용. 거래정지·상장폐지 위험이 사실상 없는 종목이어야 한다.
PROBE_TICKER = "005930"
PROBE_LOOKBACK_DAYS = 12          # 연휴를 넉넉히 덮는다
MIN_COVERAGE = 0.90               # 이 아래면 실패로 본다


def last_trading_day(today: date) -> date | None:
    """데이터가 실제로 존재하는 마지막 거래일. 공휴일 달력을 쓰지 않는다."""
    sys.path.insert(0, str(SRC))
    import warnings
    warnings.filterwarnings("ignore")
    import FinanceDataReader as fdr

    start = today - timedelta(days=PROBE_LOOKBACK_DAYS)
    try:
        df = fdr.DataReader(PROBE_TICKER, start, today)
    except Exception as e:                                    # noqa: BLE001
        print(f"거래일 탐지 실패({type(e).__name__}: {e}) — 시세 소스를 확인할 것.",
              file=sys.stderr)
        return None
    if df is None or df.empty:
        print(f"거래일 탐지 실패: {PROBE_TICKER} 가 최근 "
              f"{PROBE_LOOKBACK_DAYS}일간 데이터를 주지 않는다.", file=sys.stderr)
        return None
    return df.index[-1].date()


def coverage(as_of: date) -> tuple[int, int]:
    """(그 날짜에 적재된 종목 수, 재무가 있어 대상이 되는 종목 수)."""
    import duckdb
    con = duckdb.connect(str(REPO / "data" / "pit.duckdb"), read_only=True)
    try:
        got = con.execute("SELECT count(*) FROM prices WHERE date = ?",
                          [as_of]).fetchone()[0]
        want = con.execute("""
            SELECT count(DISTINCT m.ticker) FROM security_master m
            WHERE m.corp_code IS NOT NULL
              AND NOT coalesce(m.is_preferred, false)
              AND NOT coalesce(m.is_spac, false)
              AND m.ticker IN (SELECT DISTINCT ticker FROM facts_financial)
        """).fetchone()[0]
        return int(got), int(want)
    finally:
        con.close()


def run(cmd: list[str]) -> int:
    import os
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call([sys.executable, "-m", "pipeline.cli", *cmd],
                           cwd=str(REPO), env=env)


def log(record: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="일별 시세 갱신")
    p.add_argument("--as-of", help="YYYY-MM-DD. 생략하면 마지막 거래일 자동 탐지")
    p.add_argument("--force", action="store_true",
                   help="이미 적재된 날짜라도 다시 받는다")
    p.add_argument("--derive", action="store_true",
                   help="시세 갱신 후 파생지표까지 재산출")
    p.add_argument("--scan", action="store_true",
                   help="파생 후 스크린을 **스캔 모드**로 돌려 바스켓 표류를 보고한다. "
                        "바스켓은 갱신되지 않는다 — 확정은 `screen --rebalance`")
    p.add_argument("--screens", nargs="*", default=None,
                   help="스캔할 스크린 (기본: 등록된 전부). 한 스크린만 보려면 "
                        "예: --screens deep_value")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--source", default="fdr")
    p.add_argument("--cross-check", nargs="*",
                   help="종가를 대조할 추가 소스(예: kis). 없으면 주가 오류를 잡을 수 없다")
    a = p.parse_args()

    started = datetime.now()
    as_of = (datetime.strptime(a.as_of, "%Y-%m-%d").date() if a.as_of
             else last_trading_day(date.today()))
    if as_of is None:
        log({"at": started.isoformat(timespec="seconds"), "status": "probe_failed"})
        return 2

    if as_of != date.today():
        print(f"오늘({date.today()})은 거래일이 아니거나 장중이다 "
              f"— 마지막 거래일 {as_of} 기준으로 받는다.")

    got, want = coverage(as_of)
    # **부분 적재를 완료로 오인하면 안 된다.** 행이 있다는 것과 다 받았다는 것은
    # 다르다 — 어제 4종목만 시험 삼아 넣은 상태를 '완료'로 읽으면 그날 시세가
    # 영영 안 채워진다. 커버리지 기준을 넘겨야 건너뛴다.
    # **멱등성은 단계별로 적용한다.** 수집을 건너뛴다고 후속 단계까지 건너뛰면,
    # 수동으로 ingest-prices 를 돌린 뒤 --derive/--scan 을 붙여도 아무 일도 안 난다.
    already = bool(want) and got / want >= MIN_COVERAGE and not a.force
    if already:
        # 다시 받아도 upsert 라 결과는 같다. 다만 5분을 쓸 이유가 없다.
        print(f"수집 생략: {as_of} 는 이미 {got}/{want}종목({got / want:.1%}) 적재됨 "
              f"(--force 로 재수집)")
        rc = 0
    else:
        if got:
            print(f"{as_of} 에 {got}/{want}종목만 있다 — 부분 적재로 보고 다시 받는다.")
        cmd = ["ingest-prices", "--as-of", str(as_of), "--with-facts",
               "--workers", str(a.workers), "--source", a.source]
        if a.cross_check:
            cmd += ["--cross-check", *a.cross_check]
        rc = run(cmd)

    got, want = coverage(as_of)
    ratio = got / want if want else 0.0
    elapsed = round((datetime.now() - started).total_seconds(), 1)
    rec = {"at": started.isoformat(timespec="seconds"), "as_of": str(as_of),
           "rows": got, "expected": want, "coverage": round(ratio, 4),
           "elapsed_s": elapsed, "rc": rc, "ingested": not already}

    if rc != 0 or ratio < MIN_COVERAGE:
        # 조용히 성공으로 끝내면 다음 날 이상을 눈치채지 못한다.
        rec["status"] = "failed"
        log(rec)
        print(f"실패: rc={rc}, 커버리지 {ratio:.1%} < {MIN_COVERAGE:.0%} "
              f"({got}/{want}종목). 시세 소스를 확인할 것.", file=sys.stderr)
        return 1

    if a.derive:
        rec["derive_rc"] = run(["derive", "--as-of", str(as_of)])
        if rec["derive_rc"] != 0:
            rec["status"] = "derive_failed"
            log(rec)
            return 1

    if a.scan:
        # 스캔은 바스켓을 건드리지 않는다. 진입·이탈 후보만 보고한다.
        #
        # **등록된 스크린을 전부 돈다.** 이 시스템의 쓸모는 '종목 추천'이 아니라
        # 감시망이다 — 가설이 여러 개면 감시망도 여러 개여야 한다. 하나만 돌리면
        # 나머지 스크린의 진입·이탈은 다음 리밸런스까지 아무도 모른다.
        # 파생지표는 공유하므로 스크린을 늘려도 추가 비용은 게이트 연산뿐이다.
        sys.path.insert(0, str(SRC))
        from pipeline.screen.registry import SCREENS
        names = a.screens if a.screens is not None else sorted(SCREENS)
        rcs = {}
        for name in names:
            print(f"\n── 스캔: {name} " + "─" * 40)
            rcs[name] = run(["screen", "--as-of", str(as_of), "--screen", name])
        rec["scan_rc"] = rcs
        # 스캔 실패는 시세 수집 실패와 다르다 — 시세는 들어왔으므로 성공으로
        # 끝내되, 어느 스크린이 깨졌는지는 로그에 남긴다.
        if failed_scans := [n for n, rc in rcs.items() if rc != 0]:
            print(f"주의: 스캔 실패 {failed_scans}", file=sys.stderr)
    rec["status"] = "ok"
    rec["elapsed_s"] = round((datetime.now() - started).total_seconds(), 1)
    log(rec)
    print(f"완료: {as_of} {got}/{want}종목 ({ratio:.1%}), {rec['elapsed_s']}초"
          f"{'' if rec['ingested'] else ' (수집 생략)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
