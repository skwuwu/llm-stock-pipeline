"""LLM 테마 태깅 캐스케이드 — 싼 모델 우선, 애매한 것만 승급.

  Stage 0  후보 좁히기   무료 (키워드 + 섹터 사전확률)  전 종목
  Stage 1  1차 태깅      claude-haiku-4-5   $1/$5      전 종목
  Stage 2  승급 재판정   claude-sonnet-5    $3/$15     보통 25~35%
  Stage 3  교차 판정     claude-opus-5      $5/$25     선택, 분쟁 건만

Stage 0이 사전 37개를 종목당 후보 K개로 줄여 Stage 1의 출력 탐색 공간을 좁힌다.
Stage 2 승급 조건은 '싼 모델이 못 푼 것' + '다이제스트 상단에 실릴 것' 두 가지다.
core 배정을 승급시키는 이유는 그것이 사람이 실제로 읽는 결과이기 때문.

프롬프트 캐싱: system 규칙 + 사전 블록을 하나의 프리픽스로 묶어 cache_control 을
마지막 블록에 단다. 전 종목 호출이 동일 프리픽스를 공유하므로 캐시 읽기는 입력가의
약 0.1배. 사전 블록만으로는 Haiku 4.5의 캐시 최소 4096토큰에 미달하므로 반드시
규칙과 합쳐야 한다(미달 시 에러 없이 조용히 캐시되지 않는다).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from pipeline.themes.validate import render_taxonomy_block, resolve_sectors

PROMPT_VERSION = "tag@v1"

MODEL_CHEAP = "claude-haiku-4-5"
MODEL_DEEP = "claude-sonnet-5"
MODEL_ADJUDICATE = "claude-opus-5"

CONCURRENCY = {MODEL_CHEAP: 12, MODEL_DEEP: 6, MODEL_ADJUDICATE: 3}

CANDIDATE_K = 6          # Stage 0이 넘기는 후보 테마 수
MAX_ASSIGNMENTS = 3      # 종목당 최대 배정 테마 수
ESCALATE_CONFIDENCE = 0.75

# 샘플링을 끈다. 이 파이프라인의 산출물은 창작이 아니라 **측정값**이다 —
# 오분류율을 개선 지표로 쓰려면 같은 입력이 같은 출력을 내야 한다.
#
# 실측(2026-08-08): 온도 기본값으로 딥밸류 62종목을 재실행했더니 LLM 의 테마
# 선택은 그대로였는데(검증 전 재현율 80.3% 동일) role 판정이 흔들려 A+B
# 재현율이 80.3% → 73.8% 로 내려갔다. CJ대한통운·아이에스동서가 core→adjacent,
# 팜스토리가 reject 로 바뀐 식이다. 택소노미를 고친 직후였기 때문에
# **그 변화를 택소노미 탓으로 오귀인할 뻔했다.** 눈금이 흔들리면 개선을 잴 수 없다.
TEMPERATURE = 0.0

# **온도를 받지 않는 모델이 있다.** Sonnet 5 는 `temperature` 를 보내면
# 400 `temperature is deprecated for this model` 로 거절한다(실측).
# 그래서 심층 단계는 온도로 고정할 수 없다 — 대신 변동폭을 실측해 알고 있어야
# 한다. tests/test_m4_determinism.py 가 그 값을 기록한다.
TEMPERATURE_SUPPORTED = {MODEL_CHEAP}


# ── 구조화 출력 스키마 (자유 텍스트 금지) ────────────────────────────────
TAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme_id": {"type": "string"},
                    "role": {"type": "string", "enum": ["core", "adjacent", "peripheral"]},
                    "rationale": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                    "evidence_source": {"type": "string", "enum": ["business", "segments", "disclosures"]},
                    "revenue_share_claim": {"type": "number"},
                    "segment_name": {"type": ["string", "null"]},
                    "confidence": {"type": "number"},
                },
                "required": ["theme_id", "role", "rationale", "evidence_quote",
                             "evidence_source", "revenue_share_claim",
                             "segment_name", "confidence"],
                "additionalProperties": False,
            },
        },
        "proposed_new_theme": {"type": ["string", "null"]},
        "abstain": {"type": "boolean"},
    },
    "required": ["assignments", "proposed_new_theme", "abstain"],
    "additionalProperties": False,
}

SYSTEM_RULES = f"""\
당신은 한국·일본 상장기업의 사업 내용을 읽고 투자 테마로 분류하는 추출기다.
판단이 아니라 추출이 임무다. 아래 규칙은 예외 없이 적용된다.

1. evidence_quote 는 제공된 문서에서 **문자 그대로** 복사한다. 바꿔쓰기·요약·번역 금지.
   원문에 없는 문장을 쓰면 그 배정은 자동 폐기된다.
2. 근거를 찾을 수 없으면 assignments 를 비우고 abstain=true 로 응답한다. 추측 금지.
3. 테마는 제시된 후보 목록에서만 고른다. 최대 {MAX_ASSIGNMENTS}개. 억지로 채우지 않는다.
4. 후보 중 맞는 것이 없고 사업 내용이 명확한 새 테마에 해당하면 proposed_new_theme 에
   제안만 한다. assignments 에 임의의 id 를 만들어 넣지 않는다.
5. role 은 해당 사업이 회사 전체에서 차지하는 비중으로 정한다.
   core = 주력, adjacent = 유의미하나 부수적, peripheral = 언급 수준.
6. revenue_share_claim 은 해당 테마 관련 사업의 매출 비중 추정치(0~1)다.
   세그먼트 자료에 근거가 없으면 0 으로 둔다. 지어내지 않는다.
6-1. segment_name 에는 그 비중의 근거가 된 **보고 세그먼트 이름을 원문 표기 그대로**
   적는다(예: "DX", "중전기기"). 제시된 세그먼트 목록에 없으면 null 로 둔다.
   이 값은 실제 보고 수치와 대조되며, 없는 이름을 적으면 배정이 강등된다.
7. 주가, 밸류에이션, 투자 매력도, 목표주가, 매수/매도 의견을 언급하지 않는다.
   사업 내용과 공시 사실만으로 판단한다.
8. rationale 은 40자 이내 한 문장.

<calibration>
아래는 판정 기준을 고정하기 위한 예시다. 이 종목들을 실제 입력으로 취급하지 말 것.

예시 A — core 배정. 세그먼트에 근거가 있고 인용이 원문 그대로다.
  입력: business 에 "당사는 초고압 변압기와 가스절연개폐장치를 제조하며 북미
        전력청향 수주가 확대되고 있습니다", segments 에 {{"중전기기": 0.72}}
  판정: theme_id=ai_datacenter_power, role=core, revenue_share_claim=0.72,
        evidence_quote 는 위 문장을 한 글자도 바꾸지 않고 복사.

예시 B — peripheral 강등. 신규 진출을 발표했으나 매출 근거가 없다.
  입력: business 에 "2차전지 양극재 시장 진출을 위해 파일럿 라인을 구축 중입니다",
        segments 에 해당 항목 없음
  판정: role=peripheral, revenue_share_claim=0.0. core 로 올리지 않는다.
        진출 발표만으로 주력 사업이라고 판단하는 것이 가장 흔한 오분류다.

예시 C — abstain. 후보 테마 어디에도 해당하지 않는다.
  입력: business 가 임대업 수익만 서술하고 있고 후보 목록에 맞는 테마가 없음
  판정: assignments 를 빈 배열로 두고 abstain=true. 가장 가까운 테마를 억지로 고르지 않는다.
</calibration>
"""


# ── Stage 0: 무료 후보 좁히기 ────────────────────────────────────────────
@dataclass
class Candidate:
    theme_id: str
    score: float


def narrow_candidates(pack: dict, taxonomy: dict, sector_universe: set[str],
                      k: int = CANDIDATE_K) -> list[Candidate]:
    """키워드 히트 + 섹터 사전확률로 종목당 후보 테마를 K개로 압축.

    LLM 호출 0회. 사전 37개를 전부 프롬프트에 넣는 대신 후보만 넘기면
    출력 탐색 공간이 좁아지고 프롬프트도 짧아진다.
    """
    text = " ".join([
        pack.get("business", ""),
        json.dumps(pack.get("segments", {}), ensure_ascii=False),
        " ".join(d.get("title", "") for d in pack.get("disclosures", [])),
    ])
    sector = pack.get("sector_code")

    scored: list[Candidate] = []
    for t in taxonomy["themes"]:
        if t.get("derivation") != "llm":
            continue   # 정량 판정 테마는 LLM에 묻지 않는다
        allowed = resolve_sectors(t.get("allowed_sectors"), sector_universe)
        sector_ok = sector in allowed

        hits = sum(1 for kw in (t.get("keywords") or []) if kw and kw in text)
        if hits == 0 and not sector_ok:
            continue
        # 섹터 정합은 가산점일 뿐 필수 조건이 아니다. 신규 사업 진출은 실재하므로
        # 여기서 걸러내면 V2 검증이 판단할 기회 자체가 사라진다.
        scored.append(Candidate(t["id"], hits + (0.5 if sector_ok else 0.0)))

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:k]


# ── 캐시 ────────────────────────────────────────────────────────────────
def system_fingerprint(taxonomy: dict) -> str:
    """시스템 프롬프트의 **실제 내용** 해시.

    캐시 키를 taxonomy['version'] 같은 수기 문자열에 걸면 안 된다. 테마 정의를
    고치고 버전 올리는 걸 잊으면 재실행이 조용히 옛 결과를 돌려주고, 프롬프트를
    고쳤는데 아무것도 안 바뀌는 상황이 된다(실측: shipping_freight 제외 조항을
    추가했는데 62종목 결과가 한 건도 안 바뀌었다).
    """
    h = hashlib.sha256()
    h.update(PROMPT_VERSION.encode())
    for b in build_system(taxonomy):
        h.update(b["text"].encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cache_key(model: str, system_fp: str, pack_hash: str,
              candidates: list[str]) -> str:
    """출력에 영향을 주는 **모든 것**이 키에 들어가야 한다.

    샘플링 파라미터가 빠져 있으면 temperature 를 바꿔도 캐시가 옛 결과를
    돌려준다 — taxonomy['version'] 을 수기로 관리하다 겪은 것과 같은 고장이다
    (system_fingerprint 주석 참조). 그때는 프롬프트를 고쳤는데 62종목 결과가
    한 건도 안 바뀌었고, 이번에는 온도를 0 으로 내려도 안 바뀔 뻔했다.
    """
    h = hashlib.sha256()
    for part in (model, system_fp, pack_hash, ",".join(sorted(candidates)),
                 f"t={TEMPERATURE}"):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class TagCache:
    """동일 입력 재호출 비용 0.

    캐시가 재현성을 **만들어 주지는 않는다.** 캐시 적중은 같은 답을 돌려주지만
    그건 모델이 결정적이라서가 아니라 답을 적어뒀기 때문이다. 캐시가 빗나가는
    순간(택소노미 수정, 신규 종목) 진짜 결정성이 필요해진다 — TEMPERATURE 참조.
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> dict | None:
        p = self.root / f"{key}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def put(self, key: str, value: dict) -> None:
        (self.root / f"{key}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 프롬프트 조립 ────────────────────────────────────────────────────────
def build_system(taxonomy: dict) -> list[dict]:
    """규칙 + 사전을 하나의 캐시 프리픽스로. 순서와 바이트가 고정이어야 한다.

    cache_control 은 마지막 블록에만 단다 — 그 앞 전체가 함께 캐시된다.
    """
    return [
        {"type": "text", "text": SYSTEM_RULES},
        {"type": "text",
         "text": render_taxonomy_block(taxonomy),
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ]


def build_user(pack: dict, candidates: list[Candidate]) -> str:
    ids = ", ".join(c.theme_id for c in candidates)
    seg = pack.get("segments") or {}
    rows = seg.get("segments") if isinstance(seg, dict) else None
    segs = (", ".join(f"{r['name']}={r['share']:.1%}" for r in rows)
            if rows else (pack.get("segments_text") or "(자료 없음)"))
    titles = "\n".join(f"- {d['date']} {d['title']}" for d in pack.get("disclosures", []))
    return (
        f"<candidates>{ids}</candidates>\n\n"
        f"<company ticker=\"{pack['ticker']}\" name=\"{pack['name']}\" "
        f"sector=\"{pack.get('sector_code')}\">\n\n"
        f"<business>\n{pack.get('business', '')}\n</business>\n\n"
        # 구조화 비중은 segment_name 을 고르라고 보여주고,
        # 원문 줄은 evidence_source="segments" 로 인용할 수 있게 함께 보여준다
        # (원문이 없으면 세그먼트 인용은 V1 에서 전부 실패한다).
        f"<segments>{segs}</segments>\n"
        f"<segments_raw>{pack.get('segments_text', '')}</segments_raw>\n\n"
        f"<disclosures>\n{titles}\n</disclosures>\n"
        f"</company>"
    )


def _request_params(model: str, taxonomy: dict, pack: dict,
                    candidates: list[Candidate]) -> dict:
    params: dict[str, Any] = {
        "model": model,
        # 배정 3건 × (인용문 원문 그대로 + rationale). 인용을 요약 없이 복사하게
        # 하는 설계라 출력이 길다. 2048 은 절단돼 JSON 이 깨졌다(stop_reason=max_tokens).
        "max_tokens": 4096,
        "system": build_system(taxonomy),
        "messages": [{"role": "user", "content": build_user(pack, candidates)}],
        "output_config": {"format": {"type": "json_schema", "schema": TAG_SCHEMA}},
    }
    # effort 는 Haiku 4.5 에서 지원되지 않는다(에러). Sonnet 5 이상에만 설정.
    if model != MODEL_CHEAP:
        params["output_config"]["effort"] = "medium"
    # 온도는 반대다 — Sonnet 5 는 deprecated 로 400 을 낸다. 지원 모델에만 건다.
    if model in TEMPERATURE_SUPPORTED:
        params["temperature"] = TEMPERATURE
    return params


class LLMResponseError(RuntimeError):
    pass


def _parse(resp) -> dict:
    """응답에서 JSON 을 꺼낸다.

    next() 를 그냥 쓰면 텍스트 블록이 없을 때 StopIteration 이 나고, 코루틴
    안에서는 그게 'coroutine raised StopIteration' 이라는 원인 불명 에러로
    바뀐다. 무엇 때문에 비었는지(대개 max_tokens 절단)를 남겨야 한다.
    """
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise LLMResponseError(
            f"응답에 텍스트 블록이 없다 (stop_reason={getattr(resp, 'stop_reason', '?')}, "
            f"blocks={[b.type for b in resp.content]})")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMResponseError(
            f"JSON 파싱 실패 (stop_reason={getattr(resp, 'stop_reason', '?')}): "
            f"{text[:200]}") from e


# ── Stage 1/2 실행 ───────────────────────────────────────────────────────
def should_escalate(result: dict, pack: dict) -> tuple[bool, str]:
    """싼 모델이 못 푼 것과 결과에 실릴 것만 승급시킨다."""
    if result.get("proposed_new_theme"):
        return True, "proposed_new_theme"
    for a in result.get("assignments", []):
        if a["confidence"] < ESCALATE_CONFIDENCE:
            return True, "low_confidence"
        if a["role"] == "core":
            return True, "core_assignment"   # tier A 후보 → 다이제스트 본문
        # LLM 주장 비중과 실제 세그먼트 비중의 괴리는 값싼 정합성 신호다
        actual = _segment_share(pack, a["theme_id"])
        if actual is not None and abs(a["revenue_share_claim"] - actual) > 0.20:
            return True, "share_mismatch"
    return False, ""


def _segment_share(pack: dict, theme_id: str) -> float | None:
    return (pack.get("theme_segment_share") or {}).get(theme_id)


async def _call(client, sem: asyncio.Semaphore, params: dict) -> dict:
    async with sem:
        for attempt in range(4):
            try:
                return _parse(await client.messages.create(**params))
            except (anthropic.RateLimitError, anthropic.InternalServerError):
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
            except json.JSONDecodeError:
                # 구조화 출력이 max_tokens 로 잘린 경우 — 늘려서 1회 재시도
                if attempt >= 1:
                    raise
                params = {**params, "max_tokens": 4096}
        raise RuntimeError("unreachable")


async def tag_universe(
    packs: list[dict],
    taxonomy: dict,
    sector_universe: set[str],
    cache: TagCache,
    client: anthropic.AsyncAnthropic | None = None,
) -> list[dict]:
    """생존 종목 전체를 캐스케이드로 태깅한다."""
    client = client or anthropic.AsyncAnthropic()
    sems = {m: asyncio.Semaphore(n) for m, n in CONCURRENCY.items()}
    version = taxonomy["version"]           # 기록용(사람이 읽는 값)
    system_fp = system_fingerprint(taxonomy)   # 캐시 무효화용(내용 해시)

    async def one(pack: dict) -> dict:
        cands = narrow_candidates(pack, taxonomy, sector_universe)
        if not cands:
            return {"ticker": pack["ticker"], "abstain": True, "assignments": [],
                    "tier_model": None, "reason": "no_candidate_theme"}

        ids = [c.theme_id for c in cands]

        # Stage 1 — 싼 모델
        key = cache_key(MODEL_CHEAP, system_fp, pack["pack_hash"], ids)
        result = cache.get(key)
        if result is None:
            result = await _call(client, sems[MODEL_CHEAP],
                                 _request_params(MODEL_CHEAP, taxonomy, pack, cands))
            cache.put(key, result)
        model_used, escalated_for = MODEL_CHEAP, None

        # Stage 2 — 승급
        do_escalate, reason = should_escalate(result, pack)
        if do_escalate:
            key2 = cache_key(MODEL_DEEP, system_fp, pack["pack_hash"], ids)
            deep = cache.get(key2)
            if deep is None:
                deep = await _call(client, sems[MODEL_DEEP],
                                   _request_params(MODEL_DEEP, taxonomy, pack, cands))
                cache.put(key2, deep)
            result, model_used, escalated_for = deep, MODEL_DEEP, reason

        return {**result, "ticker": pack["ticker"], "tier_model": model_used,
                "escalated_for": escalated_for, "candidates": ids,
                "prompt_version": PROMPT_VERSION, "taxonomy_version": version,
                "system_fingerprint": system_fp}

    return await asyncio.gather(*(one(p) for p in packs))


# ── Batch API 경로 (월간 실행이면 이쪽이 맞다) ───────────────────────────
def build_batch_requests(packs: list[dict], taxonomy: dict,
                         sector_universe: set[str], model: str = MODEL_CHEAP) -> list[dict]:
    """Stage 1을 Batch API로 제출. 토큰 비용 50% 할인, 대부분 1시간 내 완료.

    딥밸류 스크린은 지연에 민감하지 않으므로 월간 실행이라면 기본 경로로 쓸 만하다.
    결과는 순서가 보장되지 않으니 반드시 custom_id 로 조인한다.
    """
    reqs = []
    for p in packs:
        cands = narrow_candidates(p, taxonomy, sector_universe)
        if not cands:
            continue
        reqs.append({"custom_id": f"tag-{p['ticker']}",
                     "params": _request_params(model, taxonomy, p, cands)})
    return reqs


def submit_batch(requests: list[dict], client: anthropic.Anthropic | None = None):
    client = client or anthropic.Anthropic()
    return client.messages.batches.create(requests=requests)


def collect_batch(batch_id: str, client: anthropic.Anthropic | None = None) -> dict[str, dict]:
    """processing_status == 'ended' 이후 호출. custom_id 로 키잉한다."""
    client = client or anthropic.Anthropic()
    out: dict[str, dict] = {}
    for r in client.messages.batches.results(batch_id):
        ticker = r.custom_id.removeprefix("tag-")
        if r.result.type == "succeeded":
            out[ticker] = _parse(r.result.message)
        else:
            out[ticker] = {"error": r.result.type, "abstain": True, "assignments": []}
    return out


# ── V1 인용 검증 (LLM 호출 0회, 가장 값싼 방어선) ────────────────────────
_NORM = re.compile(r"[\s　]+")


def verify_citation(quote: str, pack: dict, source: str) -> bool:
    """evidence_quote 가 원문에 실제로 존재하는가.

    순수 문자열 매칭으로 환각성 배정의 대부분이 여기서 죽는다.
    exact 실패 시 fuzzy 로 완화하되, rapidfuzz 미설치면 exact 만 적용한다.
    """
    haystack = {
        "business": pack.get("business", ""),
        "segments": json.dumps(pack.get("segments", {}), ensure_ascii=False),
        "disclosures": " ".join(d.get("title", "") for d in pack.get("disclosures", [])),
    }.get(source, "")

    q = _NORM.sub("", quote)
    if not q:
        return False
    if q in _NORM.sub("", haystack):
        return True
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return False
    return fuzz.partial_ratio(q, _NORM.sub("", haystack)) >= 92
