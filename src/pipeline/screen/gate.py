"""적응형 스크린 게이트 — LLM 단계 진입 종목 수를 커트라인으로 통제.

설계:
  1. 하드 가드는 절대 튜닝하지 않는다. 자본잠식·거래정지 종목이 목표 개수를
     맞추려다 통과하면 파이프라인의 존재 이유가 사라진다.
  2. 정량 필터만 tightness 스칼라 t ∈ [0,1] 로 loose↔tight 사이를 보간한다.
     각 필터가 t에 단조여야 통과 종목수 count(t)도 단조 → 이분탐색이 성립.
  3. 목표 개수에 수렴한 뒤에도 hard_cap 으로 잘라낸다(랭킹 상위 N).
  4. 해결된 임계값을 매니페스트에 기록한다. 기록하지 않으면 그 실행은 재현 불가.

임계값이 매 실행 떠다니면 바스켓이 비경제적으로 회전하므로, 직전 실행 편입
종목은 hysteresis 만큼 느슨한 t에서 평가한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd

Direction = Literal["lower_better", "higher_better"]


@dataclass(frozen=True)
class FilterSpec:
    metric: str
    direction: Direction
    loose: float
    tight: float
    exclusive_zero: bool = False   # 0 < x < threshold. 음수 PBR/PER 통과 방지.

    @property
    def key(self) -> str:
        """매니페스트용 고유 키.

        **metric 만으로는 부족하다.** PER 10~25 같은 밴드는 같은 지표에 필터
        두 개(상한·하한)로 표현되는데, metric 을 키로 쓰면 뒤엣것이 앞엣것을
        덮어써 해결된 임계값 하나가 사라진다 — 그 실행은 재현 불가가 된다.
        """
        return f"{self.metric}.{'max' if self.direction == 'lower_better' else 'min'}"

    def threshold_at(self, t: float) -> float:
        return self.loose + (self.tight - self.loose) * t

    def mask(self, df: pd.DataFrame, t: float) -> pd.Series:
        v = pd.to_numeric(df[self.metric], errors="coerce")
        thr = self.threshold_at(t)
        if self.direction == "lower_better":
            m = v <= thr
            if self.exclusive_zero:
                m &= v > 0
        else:
            m = v >= thr
            if self.exclusive_zero:
                m &= v > 0
        return m.fillna(False)


@dataclass
class GateResult:
    survivors: pd.DataFrame
    tightness: float
    resolved_thresholds: dict[str, float]
    converged: bool
    counts: dict[str, int] = field(default_factory=dict)
    why_excluded: pd.DataFrame | None = None

    def manifest(self) -> dict:
        """런 매니페스트에 그대로 직렬화. 이게 있어야 실행이 재현 가능하다."""
        return {
            "tightness": round(self.tightness, 6),
            "resolved_thresholds": {k: round(v, 6) for k, v in self.resolved_thresholds.items()},
            "converged": self.converged,
            "counts": self.counts,
        }


def _passes_all(df: pd.DataFrame, filters: list[FilterSpec], t: float) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for f in filters:
        m &= f.mask(df, t)
    return m


class MissingGuardError(RuntimeError):
    """설정이 요구한 하드 가드를 데이터가 제공하지 않는다.

    조용히 건너뛰지 않고 실패시킨다. 가드가 적용되고 있다고 믿는데 실제로는
    적용되지 않는 상태가, 가드가 아예 없는 것보다 위험하다.
    """


def apply_hard_guards(df: pd.DataFrame, exclude_flags: Iterable[str],
                      min_market_cap: float,
                      market_cap_col: str = "market_cap_used",
                      include_sectors: Iterable[str] | None = None,
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """하드 배제. 통과 집합과 탈락 사유 로그를 함께 반환한다.

    '왜 이 종목이 안 나왔나'에 답하지 못하면 스크린은 신뢰를 잃는다.

    include_sectors: 주면 **그 섹터만** 남긴다. 테마 우선 스크린용이다 —
    밸류에이션으로 좁히지 않는 대신 업종으로 좁혀 LLM 태깅 대상을 통제한다.
    섹터 매핑은 KIND 업종이라 완벽하지 않으므로(지주사가 '기타 금융업'으로
    오는 등), 좁히면 그만큼 놓친다. 놓친 수를 탈락 로그에 남긴다.
    """
    exclude_flags = list(exclude_flags)
    missing = [f for f in exclude_flags if f not in df.columns]
    if missing:
        raise MissingGuardError(
            f"하드 가드 컬럼 없음: {missing}. "
            f"설정에서 빼거나 데이터 소스를 붙일 것 — 건너뛰지 않는다."
        )

    cap_col = next((c for c in (market_cap_col, "market_cap_total", "market_cap")
                    if c in df.columns), None)
    if cap_col is None:
        raise MissingGuardError(
            f"시총 컬럼 없음 (찾은 후보: {market_cap_col}, market_cap_total, market_cap)")

    reasons: list[pd.DataFrame] = []
    keep = pd.Series(True, index=df.index)

    for flag in exclude_flags:
        hit = df[flag].fillna(False).astype(bool)
        if hit.any():
            reasons.append(pd.DataFrame({"ticker": df.loc[hit, "ticker"], "reason": flag}))
        keep &= ~hit

    if include_sectors is not None:
        allow = set(include_sectors)
        if "sector_code" not in df.columns:
            raise MissingGuardError("include_sectors 를 썼는데 sector_code 컬럼이 없다")
        off = ~df["sector_code"].isin(allow)
        if off.any():
            reasons.append(pd.DataFrame({"ticker": df.loc[off, "ticker"],
                                         "reason": "sector_not_in_scope"}))
        keep &= ~off

    cap = pd.to_numeric(df[cap_col], errors="coerce")
    # 시총 결측도 배제한다 — 값을 모르는 종목을 통과시키면 스크린이 거짓말을 한다
    small = (cap < min_market_cap) | cap.isna()
    if small.any():
        reasons.append(pd.DataFrame({"ticker": df.loc[small, "ticker"],
                                     "reason": "below_min_market_cap_or_missing"}))
    keep &= ~small

    why = pd.concat(reasons, ignore_index=True) if reasons else pd.DataFrame(columns=["ticker", "reason"])
    return df[keep].copy(), why


def guard_effectiveness(df: pd.DataFrame, exclude_flags: Iterable[str]) -> dict[str, int]:
    """가드별로 **몇 종목을 실제로 배제했는가.**

    컬럼이 있으니 가드는 '돌고' 있지만, 유니버스 구성 때문에 한 건도 배제하지
    않는 가드가 생길 수 있다. 그 상태는 조용해서 위험하다 — 가드가 지켜주고
    있다고 믿는데 실은 아무것도 안 막고 있는 것이기 때문이다.
    실측: is_preferred 는 KIND 목록이 **법인 단위**라 우선주 종목이 유니버스에
    들어오지 않아 0건이다. 가드를 빼면 소스가 종목 단위로 바뀌었을 때 뚫린다.
    빼지 말고 **0건이라는 사실을 보이게** 두는 것이 맞다.
    """
    return {f: int(df[f].fillna(False).astype(bool).sum())
            for f in exclude_flags if f in df.columns}


def composite_z(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """가중 z-score 합. 각 지표를 생존 집합 내에서 표준화한다.

    **음수 가드**: 가중치가 음수인 지표(= 낮을수록 좋음, PER·PBR)에서 값이 0 이하면
    '아주 싸다'가 아니라 **의미 없는 값**이다. 적자면 PER 이 음수가 되는데,
    그대로 표준화하면 적자 종목이 랭킹 최상위를 차지한다.
    실측: PER −2 / 5 / 50 → 점수 0.341 / 0.220 / −0.561 (음수가 1등).

    지금은 하드가드가 적자·자본잠식을 먼저 걸러 무해하지만, 가드를 손대는 순간
    터진다. 필터(exclusive_zero)가 막아주는 것과 별개로 랭킹 자체가 안전해야 한다.
    """
    score = pd.Series(0.0, index=df.index)
    for metric, w in weights.items():
        v = pd.to_numeric(df[metric], errors="coerce")
        if w < 0:
            v = v.where(v > 0)          # 0 이하는 '모름'으로 — 최고점이 되지 않게
        sd = v.std(ddof=0)
        z = pd.Series(0.0, index=df.index) if not sd or np.isnan(sd) else (v - v.mean()) / sd
        if w < 0 and z.notna().any():
            # 결측(=음수였던 것)은 최악으로 채운다. 0(중립)으로 두면
            # 값이 있는 정상 종목보다 유리해질 수 있다.
            z = z.fillna(z.max())
        score += w * z.fillna(0.0)
    return score


def resolve_target(gate_cfg: dict) -> int:
    """토큰 예산이 켜져 있으면 목표 개수를 예산에서 역산한다."""
    target = int(gate_cfg["target_count"])
    b = gate_cfg.get("budget") or {}
    if b.get("enabled"):
        per = int(b["est_tokens_per_stock"])
        target = min(target, max(1, int(b["max_input_tokens"]) // per))
    return target


def run_gate(
    eligible: pd.DataFrame,
    filters: list[FilterSpec],
    gate_cfg: dict,
    ranking_weights: dict[str, float],
    previous_members: set[str] | None = None,
) -> GateResult:
    """하드 가드를 통과한 집합에 정량 필터를 적용해 목표 개수로 수렴시킨다."""
    missing = [f.metric for f in filters if f.metric not in eligible.columns]
    if missing:
        raise MissingGuardError(
            f"필터 지표 컬럼 없음: {missing}. 설정의 metric 이름이 metrics 테이블과 다르다."
        )
    missing_rank = [m for m in ranking_weights if m not in eligible.columns]
    if missing_rank:
        raise MissingGuardError(f"랭킹 지표 컬럼 없음: {missing_rank}")

    mode = gate_cfg.get("mode", "target_count")
    hard_cap = int(gate_cfg.get("hard_cap", 100))
    previous_members = previous_members or set()

    if mode == "fixed":
        t_star, converged = 1.0, True
    elif mode == "rank_top_n":
        t_star, converged = 0.0, True   # 느슨한 적격 기준 + 랭킹 절단
    else:
        t_star, converged = _bisect_tightness(eligible, filters, gate_cfg)

    passed = _passes_all(eligible, filters, t_star)

    # 히스테리시스: 기존 편입 종목은 더 느슨한 t에서 재평가
    hys = float(gate_cfg.get("hysteresis", 0.0))
    n_retained = 0
    if hys > 0 and previous_members:
        t_exit = max(0.0, t_star - hys)
        incumbent = eligible["ticker"].isin(previous_members)
        retained = incumbent & _passes_all(eligible, filters, t_exit) & ~passed
        n_retained = int(retained.sum())
        passed = passed | retained

    survivors = eligible[passed].copy()
    survivors["composite_score"] = composite_z(survivors, ranking_weights)
    survivors = survivors.sort_values("composite_score", ascending=False)

    n_before_cap = len(survivors)
    if len(survivors) > hard_cap:
        survivors = survivors.head(hard_cap)

    return GateResult(
        survivors=survivors.reset_index(drop=True),
        tightness=t_star,
        resolved_thresholds={f.key: f.threshold_at(t_star) for f in filters},
        converged=converged,
        counts={
            "eligible": len(eligible),
            "passed_filters": n_before_cap,
            "retained_by_hysteresis": n_retained,
            "final": len(survivors),
            "truncated_by_hard_cap": max(0, n_before_cap - hard_cap),
        },
    )


def _bisect_tightness(df: pd.DataFrame, filters: list[FilterSpec],
                      gate_cfg: dict) -> tuple[float, bool]:
    """count(t)가 target에 가장 가까워지는 t를 이분탐색.

    count(t)는 t에 대해 비증가이므로, count가 target 이하가 되는 최소 t를 찾는다.
    """
    target = resolve_target(gate_cfg)
    tol = int(gate_cfg.get("tolerance", 5))
    iters = int(gate_cfg.get("bisect_iterations", 40))

    def count(t: float) -> int:
        return int(_passes_all(df, filters, t).sum())

    n_loose, n_tight = count(0.0), count(1.0)
    if n_loose <= target:        # 가장 느슨해도 목표 미달 — 더 풀 수 없다
        return 0.0, abs(n_loose - target) <= tol
    if n_tight >= target:        # 가장 빡세도 목표 초과 — 랭킹 절단에 맡긴다
        return 1.0, abs(n_tight - target) <= tol

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if count(mid) > target:
            lo = mid
        else:
            hi = mid
    return hi, abs(count(hi) - target) <= tol


def filters_from_config(cfg: dict) -> list[FilterSpec]:
    return [
        FilterSpec(
            metric=f["metric"],
            direction=f["direction"],
            loose=float(f["loose"]),
            tight=float(f["tight"]),
            exclusive_zero=bool(f.get("exclusive_zero", False)),
        )
        for f in cfg["filters"]
    ]
