"""L2 데이터 무결성 — '계산은 도는데 결과가 전부 NaN' 인 고장을 잡는다.

이 부류는 예외를 던지지 않아 테스트 수를 늘려도 잡히지 않는다. 산출물 자체를
검사해야 한다. 실측: _safe_div 의 스칼라 분모가 1행짜리 Series 로 브로드캐스트되어
payout_ratio 가 통째로 NaN 이었는데, 그때 테스트는 196개가 모두 통과하고 있었다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.derive.metrics import (CRITICAL_METRICS, MetricsIntegrityError,
                                     _safe_div, assert_metrics_sane)


# ── 근본 원인: 스칼라 분모 ───────────────────────────────────────────
def test_safe_div_broadcasts_scalar_denominator():
    """pd.Series(100.0) 은 1행짜리다. 인덱스 정렬로 첫 행만 맞고 나머지가 NaN 이 된다."""
    a = pd.Series([25.13, 34.80, 11.90, 3.77])
    out = _safe_div(a, 100.0)
    assert out.notna().all(), "스칼라 분모에서 결측이 생겼다"
    assert out.tolist() == pytest.approx([0.2513, 0.348, 0.119, 0.0377])


def test_safe_div_aligns_mismatched_index():
    a = pd.Series([10.0, 20.0], index=[5, 6])
    b = pd.Series([2.0, 4.0], index=[0, 1])
    assert _safe_div(a, b).isna().all()      # 정렬 결과 — 조용히 섞이지 않는다


def test_safe_div_zero_denominator_becomes_nan():
    """0 나눗셈은 NaN. 무한대 결과도 NaN(1/inf = 0 은 정상이라 그대로 둔다)."""
    out = _safe_div(pd.Series([1.0, np.inf, 1.0]), pd.Series([0.0, 1.0, np.inf]))
    assert out.isna().tolist() == [True, True, False]
    assert out.iloc[2] == 0.0


# ── 가드 ─────────────────────────────────────────────────────────────
def _frame(**over):
    base = {c: [1.0, 2.0] for c in CRITICAL_METRICS}
    base.update(over)
    return pd.DataFrame(base)


def test_all_nan_column_raises():
    with pytest.raises(MetricsIntegrityError, match="전부 NaN"):
        assert_metrics_sane(_frame(roic=[np.nan, np.nan]))


def test_missing_column_raises():
    df = _frame()
    with pytest.raises(MetricsIntegrityError, match="컬럼 없음"):
        assert_metrics_sane(df.drop(columns=["debt_ratio"]))


def test_empty_frame_raises():
    with pytest.raises(MetricsIntegrityError, match="비었다"):
        assert_metrics_sane(pd.DataFrame())


def test_partial_nan_is_allowed_but_recorded():
    """결측 자체는 정상이다 — 금융업 유동비율처럼. 비율을 기록해 추이를 본다."""
    rates = assert_metrics_sane(_frame(roic=[1.0, np.nan]))
    assert rates["roic"] == 0.5
    assert rates["per"] == 0.0


def test_guard_catches_the_actual_bug_shape():
    """실제 버그 형태: 첫 행만 계산되고 나머지가 NaN. 전부-NaN 이 아니라
    '거의 전부 NaN' 이므로 예외는 안 나지만 결측률로 드러나야 한다."""
    n = 100
    col = [0.25] + [np.nan] * (n - 1)
    df = pd.DataFrame({c: [1.0] * n for c in CRITICAL_METRICS})
    rates = assert_metrics_sane(df, extra=["payout_ratio"].__class__(["payout_ratio"]))\
        if False else assert_metrics_sane(df.assign(payout_ratio=col), extra=["payout_ratio"])
    assert rates["payout_ratio"] == 0.99


# ── 실제 산출물 ──────────────────────────────────────────────────────
def test_shipped_metrics_pass_integrity():
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "data/derived/metrics_2026-08-06.parquet"
    if not p.exists():
        pytest.skip("파생 산출물 없음")
    rates = assert_metrics_sane(pd.read_parquet(p))
    # 밸류에이션 핵심 지표는 결측이 20% 를 넘으면 안 된다.
    for c in ("per", "pbr", "market_cap_used"):
        assert rates[c] < 0.20, f"{c} 결측률 {rates[c]:.1%}"


# ── 고아 지표 방지 ───────────────────────────────────────────────────
def _orphans() -> list[str]:
    """계산되지만 아무도 읽지 않는 파생 수치지표.

    셋 중 하나면 '사용 중'이다:
      (a) 설정(checks/filters/ranking)이 metric·expr 로 참조
      (b) 다른 모듈이 참조
      (c) metrics.py 안에서 다른 지표를 만드는 데 쓰임 (파생 중간값)

    CRITICAL_METRICS 등재는 (c)로 세지 않는다 — 무결성 가드에 이름이 있다는
    이유로 고아가 '사용 중'으로 위장되면 이 테스트가 무의미해진다.
    """
    import re
    import yaml
    from pathlib import Path
    import pandas as pd
    from pipeline.derive.metrics import CRITICAL_METRICS

    repo = Path(__file__).resolve().parents[1]
    art = repo / "data/derived/metrics_2026-08-06.parquet"
    if not art.exists():
        return []
    m = pd.read_parquet(art)
    src = (repo / "src/pipeline/derive/metrics.py").read_text(encoding="utf-8")
    other = " ".join(
        p.read_text(encoding="utf-8") for p in (repo / "src/pipeline").rglob("*.py")
        if "derive/metrics" not in str(p).replace("\\", "/"))
    # **등록된 모든 스크린을 읽는다.** 딥밸류 하나만 보면, GARP 전용 지표
    # (op_margin, rev_growth_fy 등)가 실제로는 쓰이는데 고아로 잡힌다.
    # 반대로 스크린을 지운 뒤 남은 지표가 조용히 살아 있는 것도 여기서 걸린다.
    from pipeline.screen.registry import SCREENS

    used: set[str] = set()
    for fn in SCREENS.values():
        cfg = yaml.safe_load(
            (repo / "configs/screen" / fn).read_text(encoding="utf-8"))
        used |= {f["metric"] for f in cfg["filters"]}
        used |= set(cfg["ranking"]["weights"])
        for c in cfg.get("checks") or []:
            used |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*",
                                   str(c.get("metric") or c.get("expr") or "")))
        used |= set(cfg.get("soft_flags_report_only") or [])

    orphans = []
    for col in m.columns:
        if not pd.api.types.is_numeric_dtype(m[col]) or pd.api.types.is_bool_dtype(m[col]):
            continue
        if col in used or re.search(rf'["\']{re.escape(col)}["\']', other):
            continue
        n_def = len(re.findall(rf'df\["{re.escape(col)}"\]\s*=', src))
        n_ref = len(re.findall(rf'"{re.escape(col)}"', src)) - n_def
        if col in CRITICAL_METRICS:
            n_ref -= 1
        if n_ref <= 0:
            orphans.append(col)
    return orphans


def test_no_orphan_derived_metrics():
    """계산 비용을 쓰고 아무도 안 읽는 지표를 남기지 않는다.

    고아 지표는 단순한 낭비가 아니다 — payout_ratio 는 고아인 채로 _safe_div
    스칼라 버그를 안고 있었고, 아무도 읽지 않아 아무도 몰랐다.
    지표를 지우거나, 쓸 곳을 만들거나 둘 중 하나를 해야 한다.
    """
    orphans = _orphans()
    assert not orphans, (
        f"소비처 없는 파생 지표: {orphans}. "
        f"설정에 체크를 만들거나 metrics.py 에서 제거할 것.")


# ── 배당 역산 주가 교차검증 ──────────────────────────────────────────
def _mx(**over):
    """price_dividend_inconsistent 판정에 필요한 최소 컬럼."""
    base = {"close": [10000.0], "dps_cash": [300.0],
            "div_yield_reported_pct": [3.0]}
    base.update({k: [v] for k, v in over.items()})
    return pd.DataFrame(base)


def _flags(df):
    """metrics.py 의 판정 로직만 떼어 재현한다(빌드 전체를 돌리지 않기 위해)."""
    from pipeline.derive.metrics import (MAX_PLAUSIBLE_DIV_YIELD_PCT,
                                         MIN_YIELD_FOR_PRICE_CHECK,
                                         PRICE_DIVERGENCE_LIMIT)
    y = pd.to_numeric(df["div_yield_reported_pct"], errors="coerce")
    dps = pd.to_numeric(df["dps_cash"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    implied = np.where((dps > 0) & (y >= MIN_YIELD_FOR_PRICE_CHECK), dps / (y / 100), np.nan)
    div = close / pd.Series(implied, index=df.index)
    impossible = (y.gt(MAX_PLAUSIBLE_DIV_YIELD_PCT) | dps.gt(close)).fillna(False)
    inconsistent = ((div.gt(PRICE_DIVERGENCE_LIMIT)
                     | div.lt(1 / PRICE_DIVERGENCE_LIMIT)) & ~impossible).fillna(False)
    return impossible.iloc[0], inconsistent.iloc[0]


def test_consistent_price_and_dividend_pass():
    """DPS 300 / 수익률 3% → 역산 10,000원. 종가와 일치."""
    assert _flags(_mx()) == (False, False)


def test_price_understated_is_caught():
    """실측: 대한제분 DPS 4,000 / 3.0% = 153,846원인데 종가는 11,510원이었다.
    종가 과소 → 시총 과소 → PER·PBR 이 인위적으로 싸 보인다."""
    assert _flags(_mx(close=11510.0, dps_cash=4000.0,
                      div_yield_reported_pct=3.0)) == (False, True)


def test_absurd_reported_yield_blames_the_dividend_not_the_price():
    """아비코전자: 수익률 30%·DPS 30원(DART 가 같은 값을 두 칸에 넣었다).
    배당이 틀린 것이지 주가를 의심할 근거가 아니다 — 배제하면 안 된다."""
    impossible, inconsistent = _flags(
        _mx(close=5990.0, dps_cash=30.0, div_yield_reported_pct=30.0))
    assert impossible and not inconsistent


def test_dps_larger_than_price_blames_the_dividend():
    """와이엔텍: DPS 18억원/주 — 배당총액을 주당란에 기입한 것이다."""
    impossible, inconsistent = _flags(
        _mx(close=5910.0, dps_cash=1_809_965_900.0, div_yield_reported_pct=1.48))
    assert impossible and not inconsistent


def test_tiny_yield_is_not_checked_at_all():
    """영풍 DPS 5원 / 0.01% → 역산 5만원. 분모가 작아 오차가 증폭되므로
    검증 대상에서 뺀다. 억지로 판정하면 정상 종목이 배제된다."""
    assert _flags(_mx(close=38150.0, dps_cash=5.0,
                      div_yield_reported_pct=0.01)) == (False, False)


def test_shipped_guard_is_enforced_and_triaged():
    import yaml
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    flags = cfg["universe"]["exclude_flags"]
    assert "price_dividend_inconsistent" in flags
    # 배당 데이터 오류는 **배제하지 않는다** — 주가를 의심할 근거가 아니다.
    assert "dividend_data_impossible" not in flags


# ── 랭킹 음수 가드 ───────────────────────────────────────────────────
def test_composite_z_never_ranks_negative_per_first():
    """적자면 PER 이 음수가 되는데, 그대로 표준화하면 '아주 싸다'로 읽혀
    적자 종목이 랭킹 1위가 된다. 하드가드가 걸러주는 것과 별개로
    랭킹 자체가 안전해야 한다(가드를 손대는 순간 터진다)."""
    from pipeline.screen.gate import composite_z
    d = pd.DataFrame({"per": [-2.0, 5.0, 50.0], "pbr": [0.5] * 3,
                      "fcf_yield": [0.0] * 3})
    z = composite_z(d, {"per": -0.40, "pbr": -0.30, "fcf_yield": 0.30})
    assert z.iloc[0] == z.min(), "음수 PER 이 최하위가 아니다"
    assert z.iloc[1] == z.max(), "PER 5 가 최상위여야 한다"


def test_composite_z_keeps_negative_values_for_higher_better():
    """fcf_yield 는 음수가 정상적인 의미를 갖는다(현금 유출). 깎아내면 안 된다."""
    from pipeline.screen.gate import composite_z
    d = pd.DataFrame({"fcf_yield": [-0.10, 0.0, 0.10]})
    z = composite_z(d, {"fcf_yield": 0.30})
    assert z.iloc[0] < z.iloc[1] < z.iloc[2]


# ── 시총 정합성 (자본·매출 대비) ─────────────────────────────────────
def _cap_flag(pbr, psr, is_fin=False):
    """metrics.py 의 market_cap_inconsistent 판정을 재현한다."""
    from pipeline.derive.metrics import (MAX_PBR_FOR_CAP_CHECK,
                                         MAX_PSR_FOR_CAP_CHECK)
    return (0 < pbr < MAX_PBR_FOR_CAP_CHECK
            and 0 < psr < MAX_PSR_FOR_CAP_CHECK and not is_fin)


def test_low_margin_distributor_is_not_flagged():
    """저마진 유통·상사는 PSR 이 정상적으로 낮다 — PSR 단독 하한은 쓸 수 없다.
    실측: 서원 PSR 0.028(매출 1.78조), KG케미칼 0.031(9.37조).
    대신 이들은 PBR 이 멀쩡하다(0.30, 0.29)."""
    assert not _cap_flag(pbr=0.302, psr=0.0275)     # 서원
    assert not _cap_flag(pbr=0.288, psr=0.0314)     # KG케미칼


def test_asset_play_with_normal_psr_is_not_flagged():
    """자산주는 PBR 이 낮을 수 있으나 PSR 은 정상이다."""
    assert not _cap_flag(pbr=0.08, psr=0.45)


def test_both_axes_broken_is_flagged():
    """시총이 자본·매출 **양쪽** 대비 불가능하면 시총 자체가 틀렸다.
    실측: 포스코스틸리온 시총 277억 / 자본 3,841억 / 매출 1조 1,275억."""
    assert _cap_flag(pbr=0.072, psr=0.0246)


def test_financials_are_exempt():
    """은행·보험은 '매출' 개념이 달라 PSR 을 그대로 쓸 수 없다."""
    assert not _cap_flag(pbr=0.07, psr=0.02, is_fin=True)


def test_pbr_floor_is_tightened_but_not_reckless():
    """0.001 은 50배 느슨해 시총이 자본의 2% 인 종목도 통과시켰다.
    다만 자산주가 극단적으로 싸질 여지는 남긴다 — '불가능' 구간만 막는다."""
    from pipeline.derive.metrics import MIN_PLAUSIBLE_PBR
    assert 0.005 <= MIN_PLAUSIBLE_PBR <= 0.05


def test_two_guards_are_complementary_not_redundant():
    """배당 기반 가드는 커버리지가 46%(DPS>0 인 종목만)라 나머지를 못 본다.
    시총 정합성은 유니버스 100% 에 적용된다 — 실측 6건 중 3건이 배당 가드 밖이다."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    art = repo / "data/derived/metrics_2026-08-06.parquet"
    if not art.exists():
        pytest.skip("파생 산출물 없음")
    m = pd.read_parquet(art)
    cap = set(m[m["market_cap_inconsistent"]]["ticker"])
    div = set(m[m["price_dividend_inconsistent"]]["ticker"])
    assert cap - div, "시총 가드가 배당 가드의 부분집합이면 추가 가치가 없다"
    assert div - cap, "배당 가드도 고유하게 잡는 것이 있어야 한다"


def test_both_guards_are_enforced():
    import yaml
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (repo / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    flags = cfg["universe"]["exclude_flags"]
    assert "market_cap_inconsistent" in flags
    assert "price_dividend_inconsistent" in flags
