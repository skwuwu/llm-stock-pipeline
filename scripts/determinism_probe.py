#!/usr/bin/env python
"""같은 입력을 두 번 태깅해 재현성을 실측한다.

    python scripts/determinism_probe.py --screen deep_value --n 20

왜 필요한가
───────────
오분류율을 **개선 지표**로 쓰려면 눈금이 고정되어야 한다. 실측(2026-08-08)에서
택소노미를 고친 직후 A+B 재현율이 80.3% → 73.8% 로 내려갔는데, 그게 택소노미
탓인지 재실행 흔들림인지 구분할 방법이 없었다. 그 상태로 라벨을 쓰면 흔들리는
눈금에 자를 맞추게 된다.

**캐시를 반드시 우회한다.** 캐시가 적중하면 두 번째 실행이 공짜로 같은 답을
돌려주는데, 그건 모델이 결정적이라서가 아니라 답을 적어뒀기 때문이다.

무엇을 재는가
─────────────
  core 일치     오분류율의 계산 단위. 이게 흔들리면 지표 자체를 못 믿는다.
  판정 일치     (status, role, share_evidence) 삼중조. 등급을 없앤 뒤 리포트에
                실리는 전부다. core 가 안정적이어도 세그먼트 이름 매칭이
                흔들리면 여기서 깨질 수 있다.
  전체 일치     role·confidence 까지 포함한 완전 일치. 참고용 상한선.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from pipeline.enrich.evidence import EvidencePack  # noqa: E402
from pipeline.llm import cascade as C  # noqa: E402
from pipeline.screen.registry import resolve  # noqa: E402
from pipeline.verify.layers import SectorMatrix, verify_batch  # noqa: E402


class NoCache:
    """캐시 우회. 이걸 안 하면 두 번째 실행이 아무것도 증명하지 못한다."""

    def get(self, key): return None
    def put(self, key, value): pass


def core_set(result: dict) -> frozenset[str]:
    return frozenset(a["theme_id"] for a in (result.get("assignments") or [])
                     if a["role"] == "core")


def full_set(result: dict) -> frozenset:
    return frozenset((a["theme_id"], a["role"], round(float(a["confidence"]), 2))
                     for a in (result.get("assignments") or []))


def main() -> int:
    p = argparse.ArgumentParser(description="태깅 재현성 실측")
    p.add_argument("--screen", default="deep_value")
    p.add_argument("--as-of", default="2026-08-06")
    p.add_argument("--n", type=int, default=20, help="표본 종목 수")
    a = p.parse_args()

    paths = resolve(a.screen, REPO, REPO / "data")
    tx = yaml.safe_load((REPO / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))
    su = set(yaml.safe_load(
        (REPO / "configs/sectors/sector_map.yaml").read_text(encoding="utf-8"))["codes"])

    files = sorted(paths.enrich_dir(a.as_of).glob("*/pack.json"))[:a.n]
    if not files:
        print(f"evidence pack 없음: {paths.enrich_dir(a.as_of)}", file=sys.stderr)
        return 1
    raw = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    packs = [EvidencePack(**r).to_cascade_input() for r in raw]
    pack_by_ticker = {r["ticker"]: r for r in raw}
    sectors = {r["ticker"]: r.get("sector_code") for r in raw}

    runs = []
    for i in (1, 2):
        print(f"실행 {i}/2 — {len(packs)}종목 (캐시 우회)", flush=True)
        res = asyncio.run(C.tag_universe(packs, tx, su, NoCache()))
        vs = verify_batch(res, pack_by_ticker, sectors, SectorMatrix())
        # 판정 = (status, role, share_evidence). 등급이 없어졌으므로 이 셋이
        # 리포트에 실리는 전부다 — confidence 소수점은 더 이상 아무것도 정하지 않는다.
        verdicts = {f"{v.ticker}|{v.theme_id}":
                    (v.status, v.role, v.share_evidence) for v in vs}
        runs.append(({r["ticker"]: r for r in res}, verdicts))

    (r1, t1), (r2, t2) = runs
    tickers = sorted(r1)
    core_eq = sum(core_set(r1[t]) == core_set(r2[t]) for t in tickers)
    full_eq = sum(full_set(r1[t]) == full_set(r2[t]) for t in tickers)
    keys = set(t1) | set(t2)
    tier_eq = sum(t1.get(k) == t2.get(k) for k in keys)

    print(f"\n{'=' * 56}\n재현성 (n={len(tickers)}종목, 배정 {len(keys)}건)")
    print(f"  core 일치   {core_eq}/{len(tickers)} ({core_eq / len(tickers):.0%})"
          f"   ← 오분류율의 계산 단위")
    print(f"  판정 일치   {tier_eq}/{len(keys)} ({tier_eq / len(keys):.0%})"
          f"   ← (status, role, share_evidence)")
    print(f"  전체 일치   {full_eq}/{len(tickers)} ({full_eq / len(tickers):.0%})"
          f"   ← role·confidence 포함")

    if diffs := [t for t in tickers if core_set(r1[t]) != core_set(r2[t])]:
        print("\ncore 가 흔들린 종목 — 이건 오분류율을 직접 움직인다:")
        for t in diffs:
            print(f"  {t}  1회차={sorted(core_set(r1[t]))}  2회차={sorted(core_set(r2[t]))}")
    if td := [k for k in sorted(keys) if t1.get(k) != t2.get(k)]:
        print("\ntier 가 흔들린 배정:")
        for k in td:
            print(f"  {k}  {t1.get(k)} → {t2.get(k)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
