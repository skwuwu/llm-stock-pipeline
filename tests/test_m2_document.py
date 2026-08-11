"""사업보고서 섹션 추출 — 잘못된 섹션을 사업 설명이라고 넘기지 않는가.

pack 의 business 가 엉뚱한 섹션이면 그 종목의 테마 배정은 애초에 측정 불가다.
found_section=True 인데 내용이 틀린 게 가장 위험하다 — 정상처럼 보이기 때문이다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pipeline.ingest.dart_document import (MIN_USEFUL_SLICE, _pick_section,
                                           _SUBHEAD, _subsections,
                                           extract_business_section)

REPO = Path(__file__).resolve().parents[1]


def _doc(body: str) -> str:
    return f"<DOCUMENT><P>{body}</P></DOCUMENT>"


# ── 상호참조 배제 ────────────────────────────────────────────────────
def test_cross_reference_is_not_mistaken_for_section_start():
    """'Ⅱ. 사업의 내용 참조하시기 바랍니다' 뒤의 엉뚱한 섹션을 집으면 안 된다.

    실측: 033270 유나이티드가 진짜 26k 섹션을 두고, 참조문구 뒤에 붙은 36k 짜리
    '외부감사에 관한 사항' 을 사업 설명으로 제출했다. 후보를 길이로만 고른 탓이다.
    """
    text = ("\nII. 사업의 내용\n1. 사업의 개요\n" + "당사는 의약품 제조업체입니다. " * 40
            + "\n2. 주요 제품\n" + "제품 설명. " * 40
            + "\nIII. 재무에 관한 사항\n" + "재무. " * 10
            + "\nⅡ. 사업의 내용 참조하시기 바랍니다.\n1. 외부감사에 관한 사항\n"
            + "감사인의 명칭. " * 400)
    body, start, skipped = _pick_section(text)
    assert skipped == 1
    assert "의약품 제조업체" in body
    assert "외부감사" not in body


def test_legitimate_subsection_named_참고사항_is_not_treated_as_xref():
    """'7. 기타 참고사항' 은 정상 소제목이다. 참조 판별이 이걸 죽이면 안 된다."""
    text = ("\nII. 사업의 내용\n1. 기타 참고사항\n" + "내용. " * 50
            + "\nIII. 재무에 관한 사항\n")
    body, _, skipped = _pick_section(text)
    assert skipped == 0
    assert body is not None and "기타 참고사항" in body


# ── 괄호 접두 소제목 ─────────────────────────────────────────────────
@pytest.mark.parametrize("title", [
    "1. (제조서비스업)사업의 개요",
    "1. (금융업) 사업의 개요",
    "4. 매출 및 수주상황",
])
def test_subheading_regex_accepts_parenthesised_prefix(title):
    """제조·금융 겸영사는 DART 가 '(제조서비스업)사업의 개요' 로 쓴다.

    이 형태를 못 잡으면 해당 회사는 우선순위 소제목이 0개로 보이고,
    예산이 자금조달 실적·유형자산 명세 같은 잡동사니로 채워진다.
    """
    assert _SUBHEAD.search(title + "\n") is not None


def test_parenthesised_overview_gets_overview_priority():
    body = ("1. (제조서비스업)사업의 개요\n" + "개요 본문. " * 30
            + "\n2. 원재료 및 생산설비\n" + "원재료. " * 30)
    subs = _subsections(body)
    assert subs[0][1] == "(제조서비스업)사업의 개요"
    assert subs[0][0] == 1                       # 개요 우선순위


# ── 예산 채우기 ──────────────────────────────────────────────────────
def test_oversized_high_priority_chunk_is_truncated_not_dropped():
    """개요가 예산보다 크다고 통째로 버리고 그 자리에 저순위를 넣으면 안 된다.

    실측: 코메론·제이에스코퍼레이션의 '사업의 개요' 가 13.7k 로 예산 12k 를 넘어
    탈락하고 '원재료 및 생산설비' 가 대신 들어갔다.
    """
    body = ("1. 사업의 개요\n" + "개요다. " * 3000
            + "\n2. 원재료 및 생산설비\n" + "원재료다. " * 100)
    ex = extract_business_section(_doc(body), max_chars=8_000)
    assert "사업의 개요" in ex.text
    assert "원재료" not in ex.text
    assert any("일부만" in d for d in ex.dropped_sections)


def test_tiny_remaining_budget_does_not_produce_a_useless_sliver():
    body = ("1. 매출 및 수주상황\n" + "매출. " * 100
            + "\n2. 사업의 개요\n" + "개요. " * 3000)
    ex = extract_business_section(_doc(body), max_chars=520)
    assert len(ex.text) <= 520
    assert MIN_USEFUL_SLICE > 520                # 잘라 넣지 않고 버린 경우
    assert "사업의 개요" in ex.dropped_sections


def test_kept_chunks_are_emitted_in_document_order():
    """예산은 우선순위로 채우되 출력은 원문 순서로. 뒤죽박죽 문서를 LLM 에 주면 안 된다."""
    body = ("1. 사업의 개요\n" + "개요 본문이다. " * 30
            + "\n2. 매출 및 수주상황\n" + "매출 본문이다. " * 30)
    ex = extract_business_section(_doc(body), max_chars=1_000)
    assert ex.text.index("사업의 개요") < ex.text.index("매출 및 수주상황")


# ── 실제 문서 회귀 ───────────────────────────────────────────────────
_CACHE = REPO / "data/raw/dart_doc"
_PACKS = sorted((REPO / "data/enrich/2026-08-06").glob("*/pack.json"))


@pytest.mark.skipif(not _PACKS or not _CACHE.exists(), reason="원문 캐시 없음")
def test_every_cached_report_yields_a_business_overview():
    """캐시된 사업보고서 전건에서 발췌 앞부분에 '사업의 개요' 가 있어야 한다.

    없다면 엉뚱한 섹션을 잡았다는 뜻이다. 이 검사가 도입 시점에 4건을 잡아냈다
    (E1=자회사 발전소, 유나이티드=외부감사, KG이니시스=자금조달, LF=유형자산).
    """
    missing = []
    for p in _PACKS:
        pk = json.loads(p.read_text(encoding="utf-8"))
        f = _CACHE / f"{pk['business_meta']['rcept_no']}.txt"
        if not f.exists():
            continue
        ex = extract_business_section(f.read_text(encoding="utf-8"))
        if "사업의개요" not in ex.text[:3000].replace(" ", ""):
            missing.append(f"{pk['ticker']} {pk['name']}: "
                           f"{re.sub(r'[[:space:]]+', ' ', ex.text)[:60]}")
    assert not missing, "사업의 개요를 못 잡은 종목:\n" + "\n".join(missing)
