"""다중 스크린 — 하나의 엔진을 여러 L3 설정으로 병행 운용한다.

저평가 단독으로는 투자 아이디어가 되지 않는다. 실측(2026-08-06): GARP 40종목과
딥밸류 62종목의 교집합은 **0** 이다. 두 스크린이 같은 경로에 쓰면 뒤엣것이
앞엣것을 덮어써, 겹치지 않는다는 사실 자체를 볼 수 없게 된다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.normalize.kr import compute_growth
from pipeline.screen import registry as reg
from pipeline.screen.gate import FilterSpec, run_gate

REPO = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 8, 6)


# ── 성장률 ───────────────────────────────────────────────────────────
def _facts(rows: list[dict]) -> pd.DataFrame:
    base = {"ticker": "A", "element": "REVENUE", "statement_basis": "CFS",
            "reported_at": date(2026, 3, 31)}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_fy_growth_uses_prior_fiscal_year():
    g = compute_growth(_facts([
        {"fiscal_year": 2024, "fiscal_end_date": date(2024, 12, 31),
         "period_type": "FY", "period_months": 12, "value": 100.0},
        {"fiscal_year": 2025, "fiscal_end_date": date(2025, 12, 31),
         "period_type": "FY", "period_months": 12, "value": 125.0},
    ]), "REVENUE")
    assert g.iloc[0]["growth_fy"] == pytest.approx(0.25)


def test_quarter_growth_compares_same_cumulative_months():
    """3개월 누적을 6개월 누적과 대면 성장률이 아니라 기간 길이를 잰다."""
    g = compute_growth(_facts([
        {"fiscal_year": 2025, "fiscal_end_date": date(2025, 3, 31),
         "period_type": "CUM", "period_months": 3, "value": 50.0},
        {"fiscal_year": 2025, "fiscal_end_date": date(2025, 6, 30),
         "period_type": "CUM", "period_months": 6, "value": 120.0},
        {"fiscal_year": 2026, "fiscal_end_date": date(2026, 3, 31),
         "period_type": "CUM", "period_months": 3, "value": 60.0},
    ]), "REVENUE")
    # 최신은 3개월 누적 60 → 전년 3개월 누적 50 과 비교해야 한다(120 이 아니라)
    assert g.iloc[0]["growth_q"] == pytest.approx(0.2)


def test_negative_base_yields_no_growth_but_flags_turnaround():
    """−10 → +100 은 −1100% 가 아니다. 부호가 뒤집혀 크기 비교가 성립하지 않는다."""
    g = compute_growth(_facts([
        {"element": "OPERATING_INCOME", "fiscal_year": 2024,
         "fiscal_end_date": date(2024, 12, 31), "period_type": "FY",
         "period_months": 12, "value": -10.0},
        {"element": "OPERATING_INCOME", "fiscal_year": 2025,
         "fiscal_end_date": date(2025, 12, 31), "period_type": "FY",
         "period_months": 12, "value": 100.0},
    ]), "OPERATING_INCOME")
    r = g.iloc[0]
    assert r["growth_fy"] is None
    assert bool(r["turnaround"]) is True
    assert "fy_base_nonpositive" in r["growth_reason"]


def test_still_negative_is_not_a_turnaround():
    g = compute_growth(_facts([
        {"element": "OPERATING_INCOME", "fiscal_year": 2024,
         "fiscal_end_date": date(2024, 12, 31), "period_type": "FY",
         "period_months": 12, "value": -10.0},
        {"element": "OPERATING_INCOME", "fiscal_year": 2025,
         "fiscal_end_date": date(2025, 12, 31), "period_type": "FY",
         "period_months": 12, "value": -5.0},
    ]), "OPERATING_INCOME")
    assert bool(g.iloc[0]["turnaround"]) is False


def test_growth_keeps_one_statement_basis():
    """기준이 섞이면 회계기준 변경을 성장으로 읽는다."""
    g = compute_growth(_facts([
        {"fiscal_year": 2024, "fiscal_end_date": date(2024, 12, 31),
         "period_type": "FY", "period_months": 12, "value": 100.0,
         "statement_basis": "CFS"},
        {"fiscal_year": 2025, "fiscal_end_date": date(2025, 12, 31),
         "period_type": "FY", "period_months": 12, "value": 125.0,
         "statement_basis": "CFS"},
        {"fiscal_year": 2025, "fiscal_end_date": date(2025, 12, 31),
         "period_type": "FY", "period_months": 12, "value": 500.0,
         "statement_basis": "OFS"},
    ]), "REVENUE")
    assert g.iloc[0]["growth_fy"] == pytest.approx(0.25)   # OFS 500 을 쓰지 않았다


def test_turnaround_is_operating_income_only():
    """매출 기저가 0 이하인 건 흑자전환이 아니라 매출 데이터 결손이다."""
    src = (REPO / "src/pipeline/derive/metrics.py").read_text(encoding="utf-8")
    i = src.index('df["turnaround"]')
    line = src[i:src.index("\n", i)]
    assert "op_turnaround" in line and "rev_turnaround" not in line


# ── 밴드 필터 ────────────────────────────────────────────────────────
def test_band_thresholds_do_not_collide_in_manifest():
    """PER 10~25 는 필터 2개다. metric 을 키로 쓰면 하나가 덮여 재현 불가가 된다."""
    lo = FilterSpec("per", "higher_better", 10.0, 10.0)
    hi = FilterSpec("per", "lower_better", 25.0, 25.0)
    assert lo.key != hi.key
    df = pd.DataFrame({"ticker": list("abcd"), "per": [5.0, 15.0, 20.0, 40.0]})
    res = run_gate(df, [lo, hi], {"mode": "fixed", "hard_cap": 10}, {"per": -1.0})
    assert set(res.resolved_thresholds) == {"per.min", "per.max"}
    assert set(res.survivors["ticker"]) == {"b", "c"}


def test_garp_per_band_is_fixed_not_interpolated():
    """밴드를 보간 대상으로 두면 이분탐색이 목표 개수를 맞추려고 넓혀버린다.

    실측: loose 35/0 → tight 25/10 으로 두었더니 target 40 에서 6.0~29.0 이 됐다.
    """
    cfg = yaml.safe_load((REPO / "configs/screen/kr_garp.yaml").read_text(encoding="utf-8"))
    per = [f for f in cfg["filters"] if f["metric"] == "per"]
    assert len(per) == 2, "PER 밴드는 상한·하한 두 필터여야 한다"
    for f in per:
        assert f["loose"] == f["tight"], f"PER 밴드가 보간된다: {f}"
    assert {f["tight"] for f in per} == {10.0, 25.0}


# ── 레지스트리 ───────────────────────────────────────────────────────
def test_every_registered_screen_has_a_config():
    for name in reg.SCREENS:
        p = reg.resolve(name, REPO, REPO / "data")
        assert p.config.exists(), f"{name} 설정 파일 없음: {p.config}"


def test_unknown_screen_fails_loudly():
    with pytest.raises(reg.UnknownScreenError):
        reg.resolve("nope", REPO, REPO / "data")


def test_screen_outputs_are_namespaced():
    """섞이면 뒤에 돌린 스크린이 앞엣것을 덮어쓴다."""
    a = reg.resolve("deep_value", REPO, REPO / "data")
    b = reg.resolve("garp", REPO, REPO / "data")
    for attr in ("screen_dir", "enrich_dir", "verify_dir", "out_dir", "tags"):
        assert getattr(a, attr)(AS_OF) != getattr(b, attr)(AS_OF), attr
    assert a.basket != b.basket


def test_tag_cache_is_shared_across_screens():
    """pack_hash 키 캐시다. 가르면 같은 종목을 두 번 태깅해 비용이 두 배가 된다."""
    a = reg.resolve("deep_value", REPO, REPO / "data")
    b = reg.resolve("garp", REPO, REPO / "data")
    assert a.tag_cache == b.tag_cache


def test_metrics_are_shared_across_screens():
    """파생지표는 전 종목 대상이라 스크린을 타지 않는다 — derive 는 한 번만."""
    a = reg.resolve("deep_value", REPO, REPO / "data")
    b = reg.resolve("quality_fcf", REPO, REPO / "data")
    assert a.metrics(AS_OF) == b.metrics(AS_OF)


def test_cli_does_not_hardcode_a_screen_config():
    """하드코딩이 하나라도 남으면 --screen 이 그 커맨드에서만 조용히 무시된다."""
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    for fn in reg.SCREENS.values():
        assert fn not in src, f"cli.py 가 {fn} 을 직접 참조한다 — _paths(a).config 를 쓸 것"


# ── 골든셋 ───────────────────────────────────────────────────────────
def test_screens_without_labels_are_refused_not_scored_at_zero():
    """라벨이 없는 스크린은 조용히 0% 를 내지 말고 명확히 거절해야 한다."""
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    assert "골든셋 라벨이 없다" in src
    i = src.index("골든셋 라벨이 없다")
    assert "return 1" in src[i:i + 700]


@pytest.mark.parametrize("name", sorted(reg.SCREENS))
def test_registered_label_paths_are_not_broken(name):
    """라벨이 **없는** 스크린은 있어도 된다(theme_hunt 는 조사 목록이라 라벨이 없다).
    금지되는 건 등록해 놓고 경로가 깨진 상태다 — 그러면 golden 이 죽는다."""
    assert name in reg.GOLDEN_LABELS, f"{name} 을 GOLDEN_LABELS 에 명시할 것 (빈 목록도 선언)"
    for p in reg.resolve(name, REPO, REPO / "data").golden:
        assert p.exists(), f"{name} 라벨 경로가 깨졌다: {p}"


def test_a_ticker_is_labeled_in_exactly_one_file():
    """두 파일에 같은 티커가 있으면 어느 쪽이 정답인지 갈리고,
    라벨을 고쳐도 반영되지 않는 파일이 생긴다."""
    from pipeline.verify.golden import load_golden
    seen: dict[str, str] = {}
    for fn in {f for v in reg.GOLDEN_LABELS.values() for f in v}:
        for tk in load_golden(REPO / "tests/golden" / fn):
            assert tk not in seen, f"{tk} 가 {seen[tk]} 와 {fn} 양쪽에 있다"
            seen[tk] = fn


def test_scoring_is_restricted_to_the_measured_cohort():
    """합친 라벨을 그대로 채점하면 남의 코호트가 전부 FN 이 된다(실측 80.3%→27.2%)."""
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    assert "gold = {k: v for k, v in gold.items() if k in tagged}" in src


# ── 스크린 설정 자체의 정합성 ────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(reg.SCREENS))
def test_config_is_wellformed(name):
    cfg = yaml.safe_load(
        reg.resolve(name, REPO, REPO / "data").config.read_text(encoding="utf-8"))
    assert cfg["gate"]["target_count"] <= cfg["gate"]["hard_cap"]
    assert cfg["ranking"]["weights"], "랭킹 가중치가 비었다"
    ids = [c["id"] for c in cfg.get("checks") or []]
    assert len(ids) == len(set(ids)), f"{name}: 체크 id 중복 {ids}"
    for f in cfg["filters"]:
        assert f["direction"] in ("lower_better", "higher_better")


def test_fcf_screen_documents_why_it_allows_losses():
    """negative_earnings 를 빼는 건 의도된 선택이다. 이유가 없으면 실수와 구분되지 않는다."""
    txt = (REPO / "configs/screen/kr_quality_fcf.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(txt)
    assert "negative_earnings" not in cfg["universe"]["exclude_flags"]
    assert "negative_earnings 를 켜지 않는다" in txt
    # 배제하지 않는 대신 반드시 보이게 해야 한다.
    assert "negative_earnings" in (cfg.get("soft_flags_report_only") or [])


def test_screens_actually_produce_different_baskets():
    """설정만 갈아끼웠는데 결과가 같다면 스크린을 나눈 의미가 없다."""
    got = {}
    for name in ("deep_value", "garp", "quality_fcf"):
        p = reg.resolve(name, REPO, REPO / "data").screen_dir(AS_OF) / "survivors.parquet"
        if not p.exists():
            pytest.skip(f"{name} 스크린 산출물 없음")
        got[name] = set(pd.read_parquet(p)["ticker"])
    assert not (got["deep_value"] & got["garp"]), (
        "딥밸류와 GARP 이 겹친다 — PER 하한이 빠졌는지 확인할 것")
    for name, s in got.items():
        assert s, f"{name} 이 0종목이다"
