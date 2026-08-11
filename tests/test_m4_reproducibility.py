"""지표의 재현성 — 어느 숫자를 개선 지표로 쓸 수 있는가.

실측(2026-08-08, scripts/determinism_probe.py, 캐시 우회 2회 실행):
    core 일치  46/48  (96%)   LLM 의 core 선택은 이산 선택이라 상대적으로 안정적
                             (첫 표본은 24/24 였으나 두 번째가 22/24 — 누적해서 본다)
    판정 일치  61/75  (81%)   세그먼트 이름 매칭이 한 단계 더 끼어 더 흔들린다
    전체 일치         (38%)   confidence 소수점까지 포함

이 차이를 리포트가 드러내지 않으면, 재실행 흔들림을 개선(또는 회귀)으로
오독한다. 실제로 택소노미를 확장한 직후 A+B 재현율이 80.3% → 73.8% 로
내려갔고 하마터면 택소노미 탓으로 결론 낼 뻔했다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.verify.golden import (REPRODUCIBILITY, Scores, cores_from_tags,
                                    cores_from_verdicts, load_golden, score)

REPO = Path(__file__).resolve().parents[1]


# ── 재현성 실측값이 살아 있는가 ──────────────────────────────────────
def test_reproducibility_records_provenance():
    """숫자만 있고 출처가 없으면 다음 사람이 재측정할 수 없다."""
    for k in ("measured_at", "screen", "n_stocks", "n_assignments", "tool"):
        assert REPRODUCIBILITY.get(k), f"{k} 가 비었다"
    assert (REPO / REPRODUCIBILITY["tool"]).exists(), "재측정 도구 경로가 깨졌다"


def test_core_is_more_reproducible_than_the_verdict():
    """이 부등호가 뒤집히면 리포트 구조의 전제가 무너진다."""
    assert REPRODUCIBILITY["core_agreement"] > REPRODUCIBILITY["verdict_agreement"]
    assert REPRODUCIBILITY["verdict_agreement"] >= REPRODUCIBILITY["full_agreement"]


def test_reproducibility_is_pooled_not_a_single_run():
    """단일 실행은 양쪽으로 튄다. 첫 표본 24/24 를 '100% 재현'이라 적었다가
    두 번째 표본 22/24 에 뒤집혔다 — 그래서 표본을 누적한다."""
    assert REPRODUCIBILITY["samples"] >= 2
    assert REPRODUCIBILITY["core_agreement"] < 1.0, (
        "완전 재현이라고 주장하려면 훨씬 큰 표본이 필요하다")


def test_noise_floor_is_declared():
    """'개선'이라 부를 수 있는 최소 차이를 명시하지 않으면 잡음을 성과로 읽는다."""
    assert REPRODUCIBILITY["recall_noise_pp"] >= 1


# ── 어느 추출이 재현 가능으로 표시되는가 ─────────────────────────────
def test_scores_defaults_to_not_reproducible():
    """기본값이 True 면 새 행을 추가할 때 검증 없이 '기준 지표'가 되어버린다."""
    assert Scores(label="x").reproducible is False


def test_reproducible_flag_survives_serialization():
    s = score({}, {}, "x", reproducible=True)
    assert s.as_dict()["reproducible"] is True


# ── CLI 리포트가 두 블록으로 갈라져 있는가 ───────────────────────────
def _golden_src() -> str:
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    i = src.index("def cmd_golden")
    return src[i:src.index("\ndef ", i + 10)]


def test_report_separates_reproducible_from_not():
    s = _golden_src()
    assert "primary = [" in s and "secondary = [" in s
    assert "기준 지표" in s and "참고" in s


def test_only_cores_from_tags_is_marked_reproducible():
    """cores_from_verdicts 는 V3 강등을 타므로 재현되지 않는다.

    이걸 reproducible=True 로 올리면 흔들리는 숫자가 기준 지표 칸에 앉는다.
    """
    s = _golden_src()
    for block, expect in (("primary = [", "cores_from_tags"),
                          ("secondary = [", "cores_from_verdicts")):
        i = s.index(block)
        j = s.index("]", i)
        body = s[i:j]
        assert expect in body
        if block == "primary = [":
            assert "cores_from_verdicts" not in body, (
                "검증 기반 행이 기준 지표 블록에 있다 — 등급 흔들림을 그대로 받는다")
        else:
            assert "reproducible=True" not in body


def test_caveat_is_printed_with_the_unstable_block():
    """경고와 함께 **재측정 방법**을 알려줘야 한다. 경로는 RP['tool'] 로 보간된다."""
    s = _golden_src()
    assert "기준 지표보다 더 흔들린다" in s
    assert "RP['tool']" in s


def test_noise_floor_is_shown_next_to_the_headline_number():
    """±얼마인지 옆에 없으면, 96.1% 와 95.7% 를 다른 값으로 읽는다."""
    s = _golden_src()
    assert "recall_noise_pp" in s and "잡음이다" in s


def test_metrics_json_carries_reproducibility():
    """산출물만 열어봐도 해석이 가능해야 한다."""
    s = _golden_src()
    assert '"reproducibility": RP' in s


# ── 실제 산출물 검증 ─────────────────────────────────────────────────
def test_emitted_metrics_mark_each_row():
    p = REPO / "data/verify/deep_value/2026-08-06/golden_metrics.json"
    if not p.exists():
        pytest.skip("golden 산출물 없음")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["reproducibility"]["core_agreement"] == REPRODUCIBILITY["core_agreement"]
    rows = d["runs"]
    assert all("reproducible" in r for r in rows)
    repro = [r for r in rows if r["reproducible"]]
    assert repro, "재현 가능한 행이 하나도 없다"
    assert all("검증 전" in r["label"] or "certain" in r["label"] for r in repro)


def test_the_reproducible_row_is_the_headline_number():
    """오분류율을 인용할 때 쓰는 행이 흔들리는 행이면 안 된다."""
    p = REPO / "data/verify/deep_value/2026-08-06/golden_metrics.json"
    if not p.exists():
        pytest.skip("golden 산출물 없음")
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["runs"][0]["reproducible"] is True, "첫 행이 기준 지표여야 한다"


# ── 라벨 규율은 그대로인가 ───────────────────────────────────────────
def test_golden_labels_still_load():
    gold = load_golden(REPO / "tests/golden/kr_core_themes_v1.jsonl")
    assert len(gold) >= 60
    assert all(g.confidence in ("certain", "contested") for g in gold.values())


def test_cores_from_tags_ignores_non_core_roles():
    tags = [{"ticker": "A", "assignments": [
        {"theme_id": "x", "role": "core"},
        {"theme_id": "y", "role": "adjacent"},
        {"theme_id": "z", "role": "peripheral"},
    ]}]
    assert cores_from_tags(tags) == {"A": {"x"}}


def test_cores_from_verdicts_drops_rejects():
    v = [{"ticker": "A", "theme_id": "x", "role": "core", "status": "rejected",
          "flags": "", "share_evidence": "unavailable"},
         {"ticker": "A", "theme_id": "y", "role": "core", "status": "verified",
          "flags": "", "share_evidence": "confirmed"}]
    assert cores_from_verdicts(v) == {"A": {"y"}}
