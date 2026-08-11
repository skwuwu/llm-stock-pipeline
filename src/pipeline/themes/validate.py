"""테마 사전 검증기.

사전 자체가 파이프라인의 조인 키이므로, 사전이 깨지면 하류 검증(V2/V3)도
전부 무의미해진다. taxonomy_v1.yaml 을 바꿀 때마다 이 검사를 통과해야 한다.

    python -m pipeline.themes.validate
    python -m pipeline.themes.validate --count-tokens   # Anthropic API로 실제 토큰 측정

종료 코드: 0 = 통과(경고 허용), 1 = ERROR 존재. CI 게이트로 사용.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
TAXONOMY = REPO / "configs" / "themes" / "taxonomy_v1.yaml"
SECTOR_MAP = REPO / "configs" / "sectors" / "sector_map.yaml"

ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REQUIRED = {
    "id", "name_ko", "derivation", "axis", "definition",
    "inclusion", "exclusion", "keywords", "allowed_sectors",
    "core_revenue_share_min",
}
VALID_DERIVATION = {"llm", "quantitative"}
VALID_AXIS = {"structural", "cyclical", "secular"}

# 모델별 프롬프트 캐시 최소 프리픽스. 이보다 짧으면 cache_control을 달아도
# 조용히 캐시되지 않는다(에러 없음). 사전 블록이 전 호출에 공유되는 프리픽스이므로
# 여기 미달하면 캐스케이드의 비용 가정이 깨진다.
CACHE_MIN_TOKENS = {"claude-haiku-4-5": 4096, "claude-sonnet-5": 1024, "claude-opus-5": 512}

KEYWORD_MAX_SHARE = 3      # 한 키워드가 4개 이상 테마에 걸리면 사전필터 신호가 죽는다
JACCARD_MERGE_HINT = 0.35  # 키워드 집합 유사도 — 병합 후보 힌트
DEFINITION_MIN_CHARS = 25


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def err(self, m: str) -> None:
        self.errors.append(m)

    def warn(self, m: str) -> None:
        self.warnings.append(m)

    def note(self, m: str) -> None:
        self.info.append(m)


def _parse_sector_spec(spec) -> tuple[str, set[str]]:
    """allowed_sectors 를 (mode, codes) 로 정규화.

    'ANY'                    -> ("any", set())
    ['A', 'B']               -> ("allow", {"A","B"})
    'NOT [BANK, SECURITIES]' -> ("deny", {"BANK","SECURITIES"})
    """
    if spec is None:
        return "any", set()
    if isinstance(spec, str):
        s = spec.strip()
        if s.upper() == "ANY":
            return "any", set()
        m = re.match(r"^NOT\s*\[(.*)\]$", s, re.IGNORECASE)
        if m:
            return "deny", {c.strip() for c in m.group(1).split(",") if c.strip()}
        return "allow", {s}
    if isinstance(spec, list):
        return "allow", {str(c).strip() for c in spec}
    raise ValueError(f"allowed_sectors 형식 불명: {spec!r}")


def resolve_sectors(spec, universe: set[str]) -> set[str]:
    """섹터 스펙을 실제 허용 코드 집합으로 전개."""
    mode, codes = _parse_sector_spec(spec)
    if mode == "any":
        return set(universe)
    if mode == "deny":
        return universe - codes
    return codes


def validate(taxonomy: dict, sector_map: dict, count_tokens: bool = False) -> Report:
    rep = Report()
    themes = taxonomy.get("themes") or []
    if not themes:
        rep.err("themes 가 비어 있음")
        return rep

    universe = set(sector_map.get("codes", {}))
    if not universe:
        rep.err("sector_map.codes 가 비어 있음")
        return rep

    # 1) 스키마 · 필드 검사 ------------------------------------------------
    ids: list[str] = []
    for i, t in enumerate(themes):
        tid = t.get("id", f"<index {i}>")
        missing = REQUIRED - set(t)
        if missing:
            rep.err(f"[{tid}] 필수 필드 누락: {sorted(missing)}")
            continue
        ids.append(t["id"])

        if not ID_RE.match(t["id"]):
            rep.err(f"[{tid}] id 는 snake_case 여야 함")
        if t["derivation"] not in VALID_DERIVATION:
            rep.err(f"[{tid}] derivation 값 오류: {t['derivation']!r}")
        if t["axis"] not in VALID_AXIS:
            rep.err(f"[{tid}] axis 값 오류: {t['axis']!r}")

        if len(str(t["definition"]).strip()) < DEFINITION_MIN_CHARS:
            rep.err(f"[{tid}] definition 이 너무 짧다 — LLM 판별 기준으로 쓸 수 없음")
        if not str(t.get("exclusion") or "").strip() and t["derivation"] == "llm":
            rep.err(f"[{tid}] exclusion 이 비어 있음 — 테마 편승 오탐을 막을 수 없음")

        share = t["core_revenue_share_min"]
        if share is not None and not (0 < float(share) <= 1):
            rep.err(f"[{tid}] core_revenue_share_min 은 (0,1] 범위여야 함: {share}")

        try:
            allowed = resolve_sectors(t["allowed_sectors"], universe)
        except ValueError as e:
            rep.err(f"[{tid}] {e}")
            continue
        unknown = allowed - universe
        if unknown:
            rep.err(f"[{tid}] sector_map 에 없는 섹터 코드: {sorted(unknown)}")
        if not allowed:
            rep.err(f"[{tid}] 허용 섹터가 공집합 — 어떤 종목도 배정될 수 없음")

        kws = t["keywords"] or []
        if t["derivation"] == "llm" and not kws:
            rep.err(f"[{tid}] llm 테마인데 keywords 가 비어 있음 — Stage-0 사전필터 불가")
        if t["derivation"] == "quantitative" and kws:
            rep.warn(f"[{tid}] quantitative 테마에 keywords 가 있음 — 사용되지 않는다")

    # 2) id 유일성 --------------------------------------------------------
    for tid, n in Counter(ids).items():
        if n > 1:
            rep.err(f"id 중복: {tid} ({n}회)")

    # 3) 키워드 충돌 ------------------------------------------------------
    kw_owner: dict[str, list[str]] = defaultdict(list)
    for t in themes:
        for kw in t.get("keywords") or []:
            kw_owner[str(kw).strip()].append(t["id"])
    for kw, owners in sorted(kw_owner.items()):
        if len(owners) > KEYWORD_MAX_SHARE:
            rep.warn(f"키워드 '{kw}' 가 {len(owners)}개 테마에 공유됨 {owners} — 사전필터 변별력 저하")

    # 4) 테마 간 유사도 (병합 후보) ---------------------------------------
    llm_themes = [t for t in themes if t.get("derivation") == "llm"]
    for a, b in combinations(llm_themes, 2):
        ka, kb = set(a.get("keywords") or []), set(b.get("keywords") or [])
        if not ka or not kb:
            continue
        j = len(ka & kb) / len(ka | kb)
        if j >= JACCARD_MERGE_HINT:
            rep.warn(f"{a['id']} ↔ {b['id']} 키워드 유사도 {j:.2f} — 병합 또는 exclusion 강화 검토")

    # 5) 섹터 커버리지 ----------------------------------------------------
    # 커버되지 않는 섹터의 종목은 스크린을 통과해도 테마를 받을 수 없다.
    covered: set[str] = set()
    for t in themes:
        if t.get("derivation") != "llm":
            continue
        try:
            covered |= resolve_sectors(t.get("allowed_sectors"), universe)
        except ValueError:
            pass
    for code in sorted(universe - covered):
        rep.warn(f"섹터 {code} 를 커버하는 LLM 테마가 없음 — 해당 섹터 생존 종목은 미분류")

    # 6) 캐시 프리픽스 크기 ------------------------------------------------
    # 실제 캐시 대상은 사전 블록이 아니라 'system 규칙 + 사전' 전체다.
    prefix = _cached_prefix(taxonomy)
    measured = _count_tokens(prefix) if count_tokens else None
    n_tok = measured if measured is not None else int(len(prefix) / 2.2)
    kind = "실측" if measured is not None else "추정(한국어는 오차 큼)"
    rep.note(f"캐시 프리픽스(규칙+사전) = {n_tok:,} 토큰 [{kind}], {len(prefix):,}자")

    worst_model, worst_min = max(CACHE_MIN_TOKENS.items(), key=lambda kv: kv[1])
    margin = n_tok / worst_min
    if margin < 1.0:
        rep.err(
            f"{worst_model}: 캐시 프리픽스 {n_tok:,}토큰 < 최소 {worst_min:,}토큰 — "
            f"cache_control 을 달아도 에러 없이 조용히 캐시되지 않는다. "
            f"system 규칙을 보강하거나 캐스케이드의 싼 모델 티어를 바꿀 것."
        )
    elif margin < 1.25 and measured is None:
        rep.warn(
            f"{worst_model} 캐시 최소({worst_min:,})까지 여유 {margin:.2f}배뿐이고 "
            f"토큰 수가 추정치다. --count-tokens 로 실측해 확정할 것."
        )

    rep.note(f"테마 {len(themes)}개 (llm {len(llm_themes)}, quantitative {len(themes) - len(llm_themes)})")
    rep.note(f"섹터 커버리지 {len(covered)}/{len(universe)}")
    return rep


def render_taxonomy_block(taxonomy: dict) -> str:
    """LLM 프롬프트에 들어가는 사전 블록. 캐시 프리픽스이므로 결정론적이어야 한다."""
    lines = [f"<taxonomy version=\"{taxonomy['version']}\">"]
    for t in sorted(taxonomy["themes"], key=lambda x: x["id"]):
        if t.get("derivation") != "llm":
            continue  # 정량 판정 테마는 LLM에 노출하지 않는다
        lines.append(f"<theme id=\"{t['id']}\" name=\"{t['name_ko']}\">")
        lines.append(f"  <definition>{' '.join(str(t['definition']).split())}</definition>")
        lines.append(f"  <include>{' '.join(str(t['inclusion']).split())}</include>")
        lines.append(f"  <exclude>{' '.join(str(t['exclusion']).split())}</exclude>")
        lines.append("</theme>")
    lines.append("</taxonomy>")
    return "\n".join(lines)


def _cached_prefix(taxonomy: dict) -> str:
    """캐스케이드가 실제로 캐싱하는 프리픽스를 그대로 재구성한다.

    cascade 가 이 모듈을 import 하므로 순환을 피해 함수 안에서 import 한다.
    """
    try:
        from pipeline.llm.cascade import SYSTEM_RULES
    except ImportError:
        SYSTEM_RULES = ""
    return SYSTEM_RULES + render_taxonomy_block(taxonomy)


def _count_tokens(text: str) -> int | None:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        r = client.messages.count_tokens(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": text}],
        )
        return r.input_tokens
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    ap.add_argument("--sectors", type=Path, default=SECTOR_MAP)
    ap.add_argument("--count-tokens", action="store_true",
                    help="Anthropic count_tokens API로 사전 블록 토큰 실측")
    ap.add_argument("--strict", action="store_true", help="경고도 실패로 취급")
    a = ap.parse_args()

    taxonomy = yaml.safe_load(a.taxonomy.read_text(encoding="utf-8"))
    sectors = yaml.safe_load(a.sectors.read_text(encoding="utf-8"))
    rep = validate(taxonomy, sectors, count_tokens=a.count_tokens)

    for m in rep.info:
        print(f"  INFO  {m}")
    for m in rep.warnings:
        print(f"  WARN  {m}")
    for m in rep.errors:
        print(f" ERROR  {m}")
    print(f"\n{len(rep.errors)} error, {len(rep.warnings)} warning")

    if rep.errors or (a.strict and rep.warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
