"""설정으로 켜고 끄는 스크린 체크.

이 테스트가 지키는 것은 정확도가 아니라 **규율**이다 — 켜져 있는데 안 도는
상태, 꺼졌는데 기록 안 되는 상태, 단조가 아닌데 이분탐색에 들어가는 상태를 막는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.screen.checks import (CheckConfigError, CheckDataError, CheckSpec,
                                    apply_checks, assert_monotone, describe,
                                    enabled_gate_filters, enabled_hard_guards,
                                    enabled_soft_flags, load_checks, preview)

REPO = Path(__file__).resolve().parents[1]

DF = pd.DataFrame({
    "ticker": ["A", "B", "C", "D"],
    "sector_code": ["MACHINERY", "BANK", "CHEM", "MACHINERY"],
    "debt_ratio": [3.0, 9.0, 0.5, None],
    "roic": [0.02, 0.30, 0.12, 0.01],
    "assets": [100.0, 100.0, 100.0, 100.0],
})


def _spec(**kw):
    base = {"id": "t", "kind": "soft_flag", "enabled": True,
            "metric": "debt_ratio", "op": ">", "threshold": 2.0}
    base.update(kw)
    return [base]


# ── 설정 검증 ────────────────────────────────────────────────────────
def test_unknown_kind_rejected():
    with pytest.raises(CheckConfigError, match="kind"):
        load_checks(_spec(kind="whatever"))


def test_duplicate_id_rejected():
    with pytest.raises(CheckConfigError, match="중복"):
        load_checks(_spec() + _spec())


def test_metric_and_expr_are_mutually_exclusive():
    with pytest.raises(CheckConfigError, match="정확히 하나"):
        load_checks(_spec(expr="debt_ratio * 2"))
    with pytest.raises(CheckConfigError, match="정확히 하나"):
        load_checks([{"id": "t", "kind": "soft_flag", "op": ">", "threshold": 1}])


def test_soft_flag_requires_op_and_threshold():
    with pytest.raises(CheckConfigError, match="op"):
        load_checks([{"id": "t", "kind": "soft_flag", "metric": "roic"}])
    with pytest.raises(CheckConfigError, match="threshold"):
        load_checks([{"id": "t", "kind": "soft_flag", "metric": "roic", "op": "<"}])


def test_gate_filter_rejects_expr():
    """이분탐색은 단조성을 요구하는데 임의 식의 단조성은 보장할 수 없다."""
    with pytest.raises(CheckConfigError, match="metric 만"):
        load_checks([{"id": "t", "kind": "gate_filter", "expr": "roic * 2",
                      "direction": "higher_better", "loose": 0, "tight": 1}])


def test_gate_filter_requires_direction_and_range():
    with pytest.raises(CheckConfigError, match="direction"):
        load_checks([{"id": "t", "kind": "gate_filter", "metric": "roic",
                      "loose": 0, "tight": 1}])


# ── 평가 ─────────────────────────────────────────────────────────────
def test_hit_column_added_and_counted():
    specs = load_checks(_spec(id="high_leverage"))
    out, rep = apply_checks(DF, specs)
    assert out["high_leverage"].tolist() == [True, True, False, False]
    assert rep.hit_counts["high_leverage"] == 2
    assert rep.enabled == ["high_leverage"]


def test_excluded_sector_is_exempt():
    """은행은 예금이 부채다. 부채비율 가드를 그대로 걸면 전부 걸린다."""
    specs = load_checks(_spec(id="high_leverage", exclude_sectors=["BANK"]))
    out, rep = apply_checks(DF, specs)
    assert out["high_leverage"].tolist() == [True, False, False, False]
    assert rep.exempt_counts["high_leverage"] == 1


def test_missing_value_is_not_a_hit():
    """모르는 것을 위반으로 단정하면 데이터 결측이 곧 배제가 된다."""
    out, _ = apply_checks(DF, load_checks(_spec(id="high_leverage")))
    assert out.loc[3, "high_leverage"] is False or not out.loc[3, "high_leverage"]


def test_expression_check_evaluates():
    specs = load_checks([{"id": "acc", "kind": "soft_flag", "enabled": True,
                          "expr": "debt_ratio / assets", "op": ">",
                          "threshold": 0.02}])
    out, _ = apply_checks(DF, specs)
    assert out["acc"].tolist() == [True, True, False, False]


# ── 규율 ─────────────────────────────────────────────────────────────
def test_enabled_check_with_missing_column_raises():
    """조용히 통과시키면 가드가 도는 줄 알고 안 도는 상태가 된다 —
    가드가 아예 없는 것보다 위험하다."""
    specs = load_checks(_spec(id="x", metric="does_not_exist"))
    with pytest.raises(CheckDataError, match="does_not_exist"):
        apply_checks(DF, specs)


def test_disabled_check_with_missing_column_is_fine_but_recorded():
    specs = load_checks(_spec(id="x", metric="does_not_exist", enabled=False))
    out, rep = apply_checks(DF, specs)
    assert "x" not in out.columns
    assert rep.disabled == ["x"]          # 꺼진 것도 기록된다
    assert rep.enabled == []


def test_enabled_check_without_its_source_raises():
    specs = load_checks(_spec(id="div", metric="roic",
                              requires_source="dart_alot_matter"))
    with pytest.raises(CheckDataError, match="dart_alot_matter"):
        apply_checks(DF, specs, available_sources=set())
    apply_checks(DF, specs, available_sources={"dart_alot_matter"})   # 있으면 통과


def test_non_monotone_gate_filter_is_rejected():
    """loose/tight 방향이 direction 과 어긋나면 이분탐색이 조용히 엉뚱하게 수렴한다."""
    specs = load_checks([{"id": "bad", "kind": "gate_filter", "enabled": True,
                          "metric": "roic", "direction": "higher_better",
                          "loose": 0.30, "tight": 0.00}])       # 거꾸로
    with pytest.raises(CheckConfigError, match="단조"):
        assert_monotone(DF, specs)


def test_monotone_gate_filter_passes():
    specs = load_checks([{"id": "ok", "kind": "gate_filter", "enabled": True,
                          "metric": "roic", "direction": "higher_better",
                          "loose": 0.00, "tight": 0.30}])
    assert_monotone(DF, specs)
    assert len(enabled_gate_filters(specs)) == 1


def test_kind_routing():
    specs = load_checks([
        {"id": "g", "kind": "hard_guard", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 0.0},
        {"id": "s", "kind": "soft_flag", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 0.05},
        {"id": "f", "kind": "gate_filter", "enabled": True, "metric": "roic",
         "direction": "higher_better", "loose": 0.0, "tight": 0.1},
        {"id": "off", "kind": "hard_guard", "enabled": False, "metric": "roic",
         "op": "<", "threshold": 0.0},
    ])
    assert enabled_hard_guards(specs) == ["g"]
    assert enabled_soft_flags(specs) == ["s"]
    assert [f.metric for f in enabled_gate_filters(specs)] == ["roic"]


# ── 실제 설정 ────────────────────────────────────────────────────────
def test_shipped_config_loads_and_is_wellformed():
    cfg = yaml.safe_load(
        (REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    specs = load_checks(cfg.get("checks"))
    assert specs, "체크가 하나도 정의돼 있지 않다"
    for s in specs:
        assert s.note, f"{s.id}: note 가 없다 — 왜 이 조건인지 남길 것"
    # 부채·수익성 계열은 금융업 제외가 비어 있으면 안 된다(예금이 부채다).
    for s in specs:
        if s.metric in ("debt_ratio", "roic", "current_ratio", "accrual_ratio",
                        "cfo_to_ni"):
            assert "BANK" in s.exclude_sectors, f"{s.id}: 금융업 제외가 없다"


def test_shipped_config_defaults_to_non_invasive():
    """soft_flag 만 켜져 있어야 스크린 결과(=골든셋 대상)가 바뀌지 않는다.
    hard_guard/gate_filter 로 승격하려면 골든셋 재라벨링을 각오해야 한다."""
    cfg = yaml.safe_load(
        (REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    specs = load_checks(cfg.get("checks"))
    assert not enabled_hard_guards(specs)
    assert not enabled_gate_filters(specs)
    assert enabled_soft_flags(specs)


def test_describe_and_preview_do_not_crash_on_missing_columns():
    specs = load_checks(_spec(id="x", metric="nope", enabled=False))
    assert "nope" in describe(specs, DF)
    assert "nope" in preview(DF, specs)


# ── 게재 축 (검증 축과 분리) ─────────────────────────────────────────
def test_risk_group_required_when_polarity_is_risk():
    """같은 약점을 말하는 플래그가 개수로 중복 계산되는 것을 막기 위한 필수 필드."""
    with pytest.raises(CheckConfigError, match="risk_group"):
        load_checks(_spec(polarity="risk"))


def test_unknown_polarity_rejected():
    with pytest.raises(CheckConfigError, match="polarity"):
        load_checks(_spec(polarity="스멜"))


def test_risk_groups_counts_categories_not_flags():
    """low_roic 과 low_roe 는 같은 약점을 두 번 말한다 — 1 로 세야 한다."""
    from pipeline.screen.checks import risk_groups
    specs = load_checks([
        {"id": "low_roic", "kind": "soft_flag", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 0.05, "polarity": "risk", "risk_group": "수익성"},
        {"id": "low_roe", "kind": "soft_flag", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 0.05, "polarity": "risk", "risk_group": "수익성"},
        {"id": "high_leverage", "kind": "soft_flag", "enabled": True,
         "metric": "debt_ratio", "op": ">", "threshold": 2.0,
         "polarity": "risk", "risk_group": "재무구조"},
    ])
    out, _ = apply_checks(DF, specs)
    g = risk_groups(out, specs)
    # A: 수익성(2건) + 재무구조 = 플래그 3개지만 범주는 2개
    # B: 재무구조만, D: 수익성만(2건이지만 1범주)
    assert g.tolist() == [2, 1, 0, 1]
    flags = out.loc[0, ["low_roic", "low_roe", "high_leverage"]].astype(bool)
    assert int(flags.sum()) == 3      # 플래그는 3개인데 범주는 2개다


def test_positive_and_neutral_flags_do_not_count_as_risk():
    """고배당·자사주는 가산 신호다. 위험 점수를 올리면 안 된다."""
    from pipeline.screen.checks import risk_groups
    specs = load_checks([
        {"id": "high_dividend", "kind": "soft_flag", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 99.0, "polarity": "positive"},
        {"id": "no_dividend", "kind": "soft_flag", "enabled": True, "metric": "roic",
         "op": "<", "threshold": 99.0, "polarity": "neutral"},
    ])
    out, _ = apply_checks(DF, specs)
    assert risk_groups(out, specs).sum() == 0


def test_disabled_risk_check_is_not_counted():
    from pipeline.screen.checks import risk_groups
    specs = load_checks(_spec(id="high_leverage", enabled=False,
                              polarity="risk", risk_group="재무구조"))
    out, _ = apply_checks(DF, specs)
    assert risk_groups(out, specs).sum() == 0


def test_shipped_config_declares_polarity_for_every_check():
    """선언하지 않으면 neutral 로 떨어져 위험 신호가 조용히 게재 판정에서 빠진다."""
    import yaml
    raw = yaml.safe_load(
        (REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    for d in raw["checks"]:
        assert "polarity" in d, f"{d['id']}: polarity 가 선언되지 않았다"


def test_verification_is_not_polluted_by_risk_flags():
    """게재 축을 따로 둔 목적은 검증 판정을 건드리지 않는 것이다.
    판정에 재무 위험이 섞이면 '테마가 맞는가'와 '싸 보이는 이유가 재무에 있는가'가
    한 숫자로 뭉개진다."""
    import inspect
    from pipeline.verify import layers
    src = inspect.getsource(layers.verify_assignment)
    for leaked in ("risk_group", "low_roic", "cfo_shortfall", "dividend_", "polarity"):
        assert leaked not in src, f"검증 판정에 재무 체크가 새어들었다: {leaked}"


# ── 가드 효과 ────────────────────────────────────────────────────────
def test_guard_effectiveness_counts_per_flag():
    """가드가 '돌고 있다'와 '무언가를 막고 있다'는 다르다."""
    from pipeline.screen.gate import guard_effectiveness
    df = pd.DataFrame({"ticker": list("ABCD"),
                       "is_spac": [False] * 4,
                       "negative_earnings": [True, True, False, False]})
    hits = guard_effectiveness(df, ["is_spac", "negative_earnings", "없는컬럼"])
    assert hits == {"is_spac": 0, "negative_earnings": 2}
    assert [g for g, n in hits.items() if n == 0] == ["is_spac"]


def test_inert_guard_is_surfaced_in_shipped_config():
    """무력 가드를 설정에서 빼면 소스가 바뀌었을 때 조용히 뚫린다.
    빼지 말고 0건이라는 사실을 보이게 둔다 — 설정에 이유가 적혀 있어야 한다."""
    raw = (REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8")
    assert "is_preferred" in raw
    assert "법인 단위" in raw, "is_preferred 가 0건인 이유가 설정에 없다"


def test_payout_check_is_wired_to_the_metric_we_compute():
    """payout_ratio 는 계산만 되고 소비처가 없던 고아 지표였다."""
    cfg = yaml.safe_load(
        (REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    specs = load_checks(cfg["checks"])
    s = next(x for x in specs if x.id == "payout_excessive")
    assert s.enabled and s.metric == "payout_ratio"
    assert s.polarity == "risk" and s.risk_group == "배당지속"
