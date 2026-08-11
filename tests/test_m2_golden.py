"""골든셋 채점 로직 + 오분류율 회귀 가드."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.verify.golden import (GoldLabel, cores_from_tags,
                                    cores_from_verdicts, load_golden, score)

REPO = Path(__file__).resolve().parents[1]
RUN = "2026-08-06"


def _g(**kw):
    d = {"ticker": "0", "name": "", "core": set()}
    d.update(kw)
    return GoldLabel(**d)


def test_score_counts_tp_fp_fn():
    gold = {"A": _g(ticker="A", core={"x"}), "B": _g(ticker="B", core={"y", "z"})}
    s = score({"A": {"x", "w"}, "B": {"y"}}, gold, "t")
    assert (s.tp, s.fp, s.fn) == (2, 1, 1)
    assert s.precision == pytest.approx(2 / 3)
    assert s.recall == pytest.approx(2 / 3)
    assert s.misclassification_rate == pytest.approx(1 / 3)


def test_abstain_is_credited_only_when_gold_is_empty():
    gold = {"A": _g(ticker="A", core=set()), "B": _g(ticker="B", core={"y"})}
    s = score({"A": set(), "B": set()}, gold, "t")
    assert s.gold_empty == 1 and s.correct_abstain == 1
    assert s.fn == 1                      # B 를 놓친 건 기권이 아니라 미탐


def test_no_prediction_is_not_a_false_positive():
    """예측이 없으면 오탐이 아니라 미탐이다. 정밀도가 기권으로 부풀지 않아야 한다."""
    s = score({}, {"A": _g(ticker="A", core={"x"})}, "t")
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.precision is None            # 분모 0 — 100% 로 보고하면 거짓말이다


def test_evidence_defect_can_be_excluded():
    gold = {"A": _g(ticker="A", core={"x"}, evidence_defect=True),
            "B": _g(ticker="B", core={"x"})}
    assert score({}, gold, "t").stocks == 2
    assert score({}, gold, "t", skip_evidence_defect=True).stocks == 1


def test_cores_from_tags_takes_only_core_role():
    tags = [{"ticker": "A", "assignments": [
        {"theme_id": "x", "role": "core"},
        {"theme_id": "y", "role": "adjacent"}]}]
    assert cores_from_tags(tags) == {"A": {"x"}}


def _v(**kw):
    return {"ticker": "A", "theme_id": "x", "role": "core", "status": "verified",
            "flags": "", "share_evidence": "confirmed", **kw}


def test_cores_from_verdicts_drops_reject_and_downgrade():
    v = [_v(theme_id="x"),
         _v(theme_id="y", status="rejected"),
         _v(theme_id="z", role="adjacent")]      # 실측 비중 미달로 강등된 것
    assert cores_from_verdicts(v) == {"A": {"x"}}


def test_share_evidence_filter_narrows_to_measured_only():
    """좁히면 재현율이 떨어지는데, 그건 분류 실패가 아니라
    세그먼트를 공시하지 않는 회사가 많다는 사실이다."""
    v = [_v(theme_id="x", share_evidence="confirmed"),
         _v(theme_id="y", share_evidence="unavailable")]
    assert cores_from_verdicts(v) == {"A": {"x", "y"}}
    assert cores_from_verdicts(v, share_evidence={"confirmed"}) == {"A": {"x"}}


def test_clean_only_drops_anything_flagged():
    v = [_v(theme_id="x", flags="sector_conflict"),
         _v(theme_id="y", flags="")]
    assert cores_from_verdicts(v, clean_only=True) == {"A": {"y"}}


def test_contested_labels_carry_a_reason():
    """confidence=contested 인데 왜 갈리는지 안 적으면 표시의 의미가 없다."""
    for g in load_golden().values():
        if g.confidence == "contested":
            assert "갈리는 이유" in g.note, f"{g.ticker}: contested 사유가 없다"


def test_certain_only_scoring_excludes_contested():
    gold = {"A": GoldLabel("A", "", {"x"}, confidence="certain"),
            "B": GoldLabel("B", "", {"y"}, confidence="contested")}
    assert score({}, gold, "t").stocks == 2
    assert score({}, gold, "t", certain_only=True).stocks == 1


def test_taxonomy_gap_is_recorded_not_hidden():
    """빈 라벨은 실패가 아니라 사전 공백의 기록이다 — 이유가 적혀 있어야
    다음 사전 개편의 근거가 된다."""
    for g in load_golden().values():
        if not g.core:
            assert g.note, f"{g.ticker}: 빈 라벨인데 이유가 없다"


def test_golden_file_is_wellformed():
    gold = load_golden()
    assert len(gold) >= 60
    import yaml
    tax = yaml.safe_load(
        (REPO / "configs/themes/taxonomy_v1.yaml").read_text(encoding="utf-8"))
    known = {t["id"] for t in tax["themes"]}
    for g in gold.values():
        assert g.core <= known, f"{g.ticker} 에 사전에 없는 테마: {g.core - known}"
        assert g.note, f"{g.ticker} 라벨에 근거 메모가 없다"


# ── 회귀 가드 ────────────────────────────────────────────────────────
_ART = [REPO / "data/llm" / f"tags_{RUN}.json",
        REPO / "data/verify" / RUN / "verdicts.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _ART),
                    reason=f"{RUN} 실행 산출물 없음")
def test_misclassification_rate_does_not_regress():
    """측정 시점 실적 (골든셋 v2, taxonomy v1.1): A+B 오분류율 2.0%, 재현율 80.3%.

    여유를 두되 조용한 악화는 막는다. LLM 재실행으로 값이 바뀌면
    이 상수를 갱신하고 왜 바뀌었는지 커밋 메시지에 남길 것.
    """
    import pandas as pd
    tags = json.loads(_ART[0].read_text(encoding="utf-8"))
    verdicts = pd.read_parquet(_ART[1]).to_dict("records")
    gold = load_golden()
    before = score(cores_from_tags(tags), gold, "before")
    passed = score(cores_from_verdicts(verdicts, tiers={"A", "B"}), gold, "A+B")

    assert passed.misclassification_rate <= 0.05, \
        f"오분류율 악화: {passed.misclassification_rate:.1%}"
    assert passed.recall >= 0.70, f"재현율 악화: {passed.recall:.1%}"

    # 지표가 '판정이 갈리는 라벨' 에 기대고 있으면 안 된다. certain-only 가 전체보다
    # 크게 나쁘면, 숫자가 데이터 품질이 아니라 라벨러 판단을 재고 있다는 뜻이다.
    certain = score(cores_from_verdicts(verdicts, tiers={"A", "B"}), gold,
                    "certain", certain_only=True)
    assert certain.precision >= passed.precision - 0.02, (
        f"certain-only 정밀도가 크게 낮다: {passed.precision:.1%} -> "
        f"{certain.precision:.1%} — 지표가 contested 라벨에 기대고 있다")

    # 검증 레이어가 정밀도를 **깎으면** 안 된다. 티어 정의를 고치기 전엔 실제로 깎고
    # 있었다(94.3% → 92.9%). '반드시 올려야 한다'로 걸지 않는 이유: 원본 출력에
    # 오탐이 0건이면 잡을 것이 없어 동률이 정상이고, 그때 실패시키면 테스트가
    # 한 번의 측정에 과적합된다.
    assert passed.precision >= before.precision, (
        f"검증이 정밀도를 떨어뜨림: {before.precision:.1%} -> {passed.precision:.1%}")
    # 그 대가로 정답을 깎아내면 안 된다.
    assert passed.recall >= before.recall - 0.10, (
        f"검증이 재현율을 과하게 깎음: {before.recall:.1%} -> {passed.recall:.1%}")


@pytest.mark.skipif(not all(p.exists() for p in _ART),
                    reason=f"{RUN} 실행 산출물 없음")
def test_tier_band_matches_flag_signal():
    """A+B 통과 = 검증실패 0건. 등급이 플래그 신호와 어긋나면 등급이 거짓말을 한다."""
    import pandas as pd
    verdicts = pd.read_parquet(_ART[1]).to_dict("records")
    assert (cores_from_verdicts(verdicts, tiers={"A", "B"})
            == cores_from_verdicts(verdicts, require_clean=True))
