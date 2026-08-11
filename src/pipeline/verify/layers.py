"""검증 레이어 — LLM 출력은 여기를 통과하기 전엔 후보(candidate)일 뿐이다.

  V1 인용 근거 검증   evidence_quote 가 원문에 실제로 있는가      → rejected (LLM 호출 0회)
  V2 섹터 정합성      테마 허용 섹터에 종목 섹터가 있는가          → flag
  V3 역할·비중        실측 매출 비중이 테마 하한을 넘는가          → role 강등
  종합               status = verified | rejected

**등급(A/B/C)을 쓰지 않는다.** 폐기하기 전 마지막 형태는 다음 문제를 안고 있었다:

  · A 와 B 를 가르는 유일한 기준이 LLM 이 자기 신고한 confidence >= 0.75 였다.
    모델이 스스로 매긴 소수점이 등급을 정하는 셈이라 검증이라 부르기 어렵다.
  · share_overclaim(LLM 주장 vs 실측 괴리)이 C 로 내리는 하드 플래그였다.
    주장값이 흔들리면 등급이 뒤집힌다 — 실측 재실행 tier 일치율 84%,
    CJ대한통운은 A↔C 를 왕복했다.
  · 그 결과 "이 종목은 B 다" 가 무엇을 뜻하는지 말할 수 없었다.

지금은 두 가지 사실만 낸다:
  status         원문 인용으로 뒷받침되는가 (아니면 rejected)
  share_evidence 매출 비중을 무엇으로 확인했는가 (confirmed/below_floor/미확인)
비중 수치는 강등 근거로만 쓰고 **숫자 그대로 리포트에 싣는다.** 판단은 사람이 한다.
재무 위험은 완전히 다른 축(risk_groups)이며 여기서 섞지 않는다.

V2 가 rejected 가 아닌 이유: 화학사의 2차전지 소재 전환처럼 진짜 신규 진출이 실재한다.

**한 테마가 여러 세그먼트에 걸칠 수 있다.** JYP 는 음반 31.4% + 매니지먼트 68.6%
가 둘 다 media_content_ip 인데, 세그먼트별로 따로 판정하면 각각 하한 0.35 에
미달해 **양쪽 다 강등**된다. 합치면 100% 다. verify_batch 가 (ticker, theme_id)
로 묶어 세그먼트를 합산한 뒤 한 번만 판정한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
_NORM = re.compile(r"[\s　]+")

FUZZY_THRESHOLD = 92
EVIDENCE_REQUIRED_ON_CONFLICT = 2
SHARE_CLAIM_TOLERANCE = 0.20   # LLM 주장 비중과 실제 비중의 허용 괴리(표시용)

VERIFIED, REJECTED = "verified", "rejected"

# share_evidence — 매출 비중을 **무엇으로** 확인했는가. 등급이 아니라 사실 서술이다.
SHARE_CONFIRMED = "confirmed"        # 실측 세그먼트 수치가 하한 이상
SHARE_BELOW_FLOOR = "below_floor"    # 실측 수치가 하한 미만 → role 강등
SHARE_NOT_FOUND = "not_found"        # 세그먼트 표는 있는데 그 이름이 없다
SHARE_UNVERIFIED = "unverified"      # 비율표만 있고 뒷받침 수치가 없다
SHARE_UNAVAILABLE = "unavailable"    # 세그먼트 자체가 없다
SHARE_NOT_CLAIMED = "not_claimed"    # core 인데 비중을 말하지 않았다

_ROLE_RANK = {"core": 2, "adjacent": 1, "peripheral": 0}


@dataclass
class Verdict:
    ticker: str
    theme_id: str
    role: str
    status: str                     # verified | rejected
    flags: list[str] = field(default_factory=list)
    reject_reason: str | None = None
    confidence: float = 0.0
    rationale: str = ""
    evidence_quote: str = ""
    actual_share: float | None = None   # 보고서에서 확인된 실제 비중(합산 후)
    claimed_share: float | None = None  # LLM 주장. 강등 근거로 쓰지 않고 표시만
    share_evidence: str = SHARE_UNAVAILABLE
    # 이 테마가 덮는 세그먼트 이름들. 하나가 아닐 수 있다 — 모듈 docstring 참조.
    segments: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.status == VERIFIED


def _norm(s: str) -> str:
    return _NORM.sub("", s or "")


# ── V1 인용 근거 검증 ────────────────────────────────────────────────
def verify_citation(quote: str, pack: dict, source: str) -> bool:
    """가장 값싸고 가장 강력한 방어선. 순수 문자열 매칭, LLM 호출 0회."""
    haystack = {
        "business": pack.get("business", ""),
        "segments": _segments_text(pack),
        "disclosures": " ".join(d.get("title", "") for d in pack.get("disclosures", [])),
    }.get(source, "")

    q = _norm(quote)
    if len(q) < 8:            # 너무 짧은 인용은 우연히 맞을 수 있다
        return False
    if q in _norm(haystack):
        return True
    # rapidfuzz 부재를 조용히 False 로 흘리면 완전일치만 남아 환각률이 부풀려진다.
    # 검증기가 "검증 못 함"을 "검증 실패"로 보고하는 일은 없어야 한다.
    try:
        from rapidfuzz import fuzz
    except ImportError as e:
        raise RuntimeError(
            "rapidfuzz 가 없어 V1 퍼지 인용 매칭을 할 수 없다. "
            "pip install rapidfuzz — 없이 돌리면 환각률이 과대 측정된다.") from e
    return fuzz.partial_ratio(q, _norm(haystack)) >= FUZZY_THRESHOLD


def verify_citation_anywhere(quote: str, pack: dict) -> str | None:
    """출처를 잘못 적었을 뿐 원문에는 있는 경우를 구제한다. 맞은 출처명을 돌려준다."""
    for src in ("business", "segments", "disclosures"):
        if verify_citation(quote, pack, src):
            return src
    return None


def _segments_text(pack: dict) -> str:
    """인용 대조용 **원문**. 구조화된 segments(수치)와 혼동하면 안 된다 —
    수치 딕셔너리를 문자열로 이어붙여 대조하면 인용 검증이 무의미해진다."""
    t = pack.get("segments_text")
    if t:
        return str(t)
    seg = pack.get("segments") or {}
    if isinstance(seg, dict):
        if "segments" in seg:                       # 구조화 형태
            return str(seg.get("source_line", ""))
        return "\n".join(str(v) for v in seg.values())   # 구버전 팩 호환
    return str(seg)


def _has_segments(pack: dict) -> bool:
    """V3 가 쓸 수 있는 세그먼트가 있는가 — **입증된 것만** 센다.

    뒷받침 없는 비율표로 core 를 강등하면 검증이 아니라 추측이 된다.
    """
    seg = pack.get("segments") or {}
    return bool(isinstance(seg, dict) and seg.get("segments")
                and seg.get("corroborated"))


def _has_uncorroborated_segments(pack: dict) -> bool:
    """수치는 뽑혔지만 뒷받침이 없는 경우(비율표). '없음'과 구분해서 남긴다."""
    seg = pack.get("segments") or {}
    return bool(isinstance(seg, dict) and seg.get("segments")
                and not seg.get("corroborated"))


def _actual_share(pack: dict, segment_names) -> float | None:
    """이 테마가 덮는 세그먼트들의 **합산** 비중. 하나도 못 찾으면 None(0 아님).

    합산하는 이유는 모듈 docstring 참조 — 한 테마가 여러 세그먼트에 걸치면
    개별 비교로는 둘 다 하한에 미달해 함께 강등된다(JYP 실측).
    이름이 중복돼도 한 번만 센다. 같은 세그먼트를 두 번 더하면 비중이 부풀어
    없는 근거를 만들어내게 된다.
    """
    if isinstance(segment_names, str):
        segment_names = [segment_names]
    names = [n for n in dict.fromkeys(segment_names or []) if n]
    if not names or not _has_segments(pack):
        return None
    from pipeline.enrich.segments import SegmentSet
    ss = SegmentSet.from_dict(pack["segments"])
    found = [s for s in (ss.share_of(n) for n in names) if s is not None]
    return sum(found) if found else None


# ── V2 섹터 정합성 ───────────────────────────────────────────────────
class SectorMatrix:
    def __init__(self, taxonomy_path: Path | None = None,
                 sector_path: Path | None = None):
        from pipeline.themes.validate import resolve_sectors
        tax = yaml.safe_load((taxonomy_path or
                              REPO / "configs/themes/taxonomy_v1.yaml").read_text("utf-8"))
        sec = yaml.safe_load((sector_path or
                              REPO / "configs/sectors/sector_map.yaml").read_text("utf-8"))
        universe = set(sec["codes"])
        self.allowed = {t["id"]: resolve_sectors(t.get("allowed_sectors"), universe)
                        for t in tax["themes"]}
        self.core_min = {t["id"]: t.get("core_revenue_share_min") for t in tax["themes"]}
        self.known = set(self.allowed)

    def conflicts(self, theme_id: str, sector_code: str | None) -> bool:
        if theme_id not in self.allowed:
            return True                     # 사전에 없는 테마 = 환각
        if not sector_code:
            return False                    # 섹터 미상이면 판정하지 않는다(플래그는 별도)
        return sector_code not in self.allowed[theme_id]


# ── 종합 판정 ────────────────────────────────────────────────────────
def verify_assignment(assignment: dict, pack: dict, matrix: SectorMatrix,
                      sector_code: str | None) -> Verdict:
    """배정 하나(또는 합쳐진 하나)를 판정한다.

    `assignment` 의 `segment_name` 은 문자열 또는 문자열 목록 둘 다 받는다 —
    verify_batch 가 같은 테마의 세그먼트를 모아 목록으로 넘긴다.
    """
    theme_id = assignment.get("theme_id", "")
    role = assignment.get("role", "peripheral")
    conf = float(assignment.get("confidence") or 0.0)
    quote = assignment.get("evidence_quote", "")
    seg_raw = assignment.get("segment_name")
    seg_names = [seg_raw] if isinstance(seg_raw, str) else list(seg_raw or [])
    seg_names = [n for n in dict.fromkeys(seg_names) if n]

    v = Verdict(ticker=pack["ticker"], theme_id=theme_id, role=role,
                status=VERIFIED, confidence=conf,
                rationale=assignment.get("rationale", ""),
                evidence_quote=quote, segments=seg_names)

    # ── 폐기 사유는 둘뿐이다 ──────────────────────────────────────
    if theme_id not in matrix.known:
        v.status, v.reject_reason = REJECTED, "unknown_theme"
        return v

    src = assignment.get("evidence_source", "business")
    if not verify_citation(quote, pack, src):
        actual_src = verify_citation_anywhere(quote, pack)
        if actual_src is None:
            v.status, v.reject_reason = REJECTED, "hallucinated_citation"
            return v
        v.flags.append(f"citation_source_mismatch:{src}->{actual_src}")

    # ── 나머지는 전부 '표시'다 ────────────────────────────────────
    if matrix.conflicts(theme_id, sector_code):
        v.flags.append("sector_conflict")
    if not sector_code:
        v.flags.append("sector_unknown")

    # V3 — **LLM 의 주장이 아니라 보고된 수치로 판정한다.**
    # 주장만 보고 하한과 비교하면 LLM 을 LLM 으로 검증하는 셈이라 아무것도 못 잡는다.
    claim = float(assignment.get("revenue_share_claim") or 0.0)
    v.claimed_share = claim or None
    floor = matrix.core_min.get(theme_id)
    actual = _actual_share(pack, seg_names)

    if actual is not None:
        v.actual_share = actual
        # 주장과 실측의 괴리는 **기록만** 한다. 강등 근거로 쓰면 LLM 이 뱉은
        # 연속값이 판정을 흔든다(실측: 재실행 tier 일치율 84%).
        if abs(actual - claim) > SHARE_CLAIM_TOLERANCE:
            v.flags.append(f"share_claim_gap:claim={claim:.2f}/actual={actual:.2f}")
        if role == "core" and floor is not None and actual < floor:
            # 이건 유지한다. **실측치와 택소노미 상수의 비교**라 LLM 주장이
            # 끼지 않고, LABELING.md R1 이 쓰는 하한과 같은 규칙이다.
            v.role = "adjacent" if actual >= floor / 2 else "peripheral"
            v.share_evidence = SHARE_BELOW_FLOOR
            v.flags.append(
                f"role_downgraded_by_actual:core->{v.role}(actual={actual:.2f}<{floor})")
        else:
            v.share_evidence = SHARE_CONFIRMED
    else:
        if _has_segments(pack):
            v.share_evidence = SHARE_NOT_FOUND
            v.flags.append(f"segment_not_found:{','.join(seg_names)}" if seg_names
                           else "segment_name_missing")
        elif _has_uncorroborated_segments(pack):
            v.share_evidence = SHARE_UNVERIFIED
            v.flags.append("segment_data_unverified")
        else:
            v.share_evidence = SHARE_UNAVAILABLE
            v.flags.append("segment_data_unavailable")
        # 비중을 **말하지 않은 것**(claim=0.0)과 **낮다고 말한 것**은 다르다.
        # 미기재를 0% 로 읽어 강등하면 검증이 아니라 메타데이터 누락 처벌이 된다.
        if role == "core" and floor is not None and 0.0 < claim < floor:
            v.role = "adjacent"
            v.flags.append(f"role_downgraded:core->adjacent(claim={claim:.2f}<{floor})")
        if claim == 0.0 and role == "core":
            v.share_evidence = SHARE_NOT_CLAIMED
            v.flags.append("core_without_share_evidence")

    return v


def merge_assignments(assignments: list[dict]) -> list[dict]:
    """같은 테마에 배정된 항목들을 하나로 합친다. **세그먼트는 목록이 된다.**

    LLM 은 세그먼트 하나당 배정 하나를 내므로, 한 테마가 두 세그먼트에 걸치면
    배정이 둘로 쪼개져 나온다. 그대로 판정하면 각각이 하한에 미달해 **둘 다**
    강등된다 — JYP 는 음반 31.4% / 매니지먼트 68.6% 로 나뉘어 합계 100% 인데도
    양쪽 다 core→adjacent 로 내려갔다.

    합치는 규칙:
      role        가장 강한 것 (core > adjacent > peripheral)
      confidence  최대
      claim       합 (서로 다른 세그먼트의 비중이므로 더하는 것이 맞다)
      quote       가장 긴 것 — V1 은 길수록 우연 일치 가능성이 낮다
      segment     합집합. 순서 유지
    """
    by_theme: dict[str, dict] = {}
    for a in assignments:
        tid = a.get("theme_id", "")
        cur = by_theme.get(tid)
        seg = a.get("segment_name")
        segs = [seg] if isinstance(seg, str) else list(seg or [])
        if cur is None:
            by_theme[tid] = {**a, "segment_name": [s for s in segs if s]}
            continue
        if _ROLE_RANK.get(a.get("role"), -1) > _ROLE_RANK.get(cur.get("role"), -1):
            cur["role"] = a.get("role")
        cur["confidence"] = max(float(cur.get("confidence") or 0.0),
                                float(a.get("confidence") or 0.0))
        cur["revenue_share_claim"] = (float(cur.get("revenue_share_claim") or 0.0)
                                      + float(a.get("revenue_share_claim") or 0.0))
        if len(a.get("evidence_quote") or "") > len(cur.get("evidence_quote") or ""):
            cur["evidence_quote"] = a.get("evidence_quote")
            cur["evidence_source"] = a.get("evidence_source", "business")
        for s in segs:
            if s and s not in cur["segment_name"]:
                cur["segment_name"].append(s)
        if (r := a.get("rationale")) and r not in (cur.get("rationale") or ""):
            cur["rationale"] = f"{cur.get('rationale', '')} / {r}".strip(" /")[:200]
    return list(by_theme.values())


def verify_batch(tag_results: list[dict], packs: dict[str, dict],
                 sectors: dict[str, str | None],
                 matrix: SectorMatrix | None = None) -> list[Verdict]:
    matrix = matrix or SectorMatrix()
    out: list[Verdict] = []
    for res in tag_results:
        tk = res.get("ticker")
        pack = packs.get(tk)
        if pack is None:
            continue
        if res.get("abstain") or not res.get("assignments"):
            continue
        for a in merge_assignments(res["assignments"]):
            out.append(verify_assignment(a, pack, matrix, sectors.get(tk)))
    return out


def verification_metrics(verdicts: list[Verdict], n_stocks: int) -> dict:
    """매 실행 기록해 시계열로 관리한다. 오분류율의 대리 지표."""
    n = len(verdicts)
    if n == 0:
        return {"assignments": 0, "stocks": n_stocks}
    rejects = [v.reject_reason for v in verdicts if v.reject_reason]
    flat = [f.split(":")[0] for v in verdicts for f in v.flags]
    ver = [v for v in verdicts if v.verified]
    return {
        "stocks": n_stocks,
        "assignments": n,
        "status": {VERIFIED: len(ver), REJECTED: n - len(ver)},
        # core 는 강등 **후** 기준이다. LLM 이 core 라고 말한 수가 아니라
        # 실측 비중이 하한을 넘어 core 로 남은 수다.
        "verified_core": sum(1 for v in ver if v.role == "core"),
        # 비중 근거의 분포. '확인됨'이 적다면 그건 실패가 아니라
        # 세그먼트를 공시하지 않는 회사가 많다는 사실이다 — 감추지 않는다.
        "share_evidence": {k: sum(1 for v in ver if v.share_evidence == k)
                           for k in sorted({v.share_evidence for v in ver})},
        "hallucinated_citation_rate": round(
            rejects.count("hallucinated_citation") / n, 4),
        "reject_reasons": {r: rejects.count(r) for r in set(rejects)},
        "flags": {f: flat.count(f) for f in set(flat)},
    }
