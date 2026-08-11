"""설정으로 켜고 끄는 스크린 체크.

한 체크 = 한 블록. `enabled` 로 on/off, `threshold` 로 절대 수치를 조정한다.
코드 수정 없이 설정만으로 다음 세 가지 역할 사이를 옮길 수 있어야 한다:

  hard_guard   절대 배제. 튜닝되지 않는다.
  gate_filter  tightness t 로 loose↔tight 보간. 종목 수 조절에 참여한다.
  soft_flag    통과시키되 표기만. 다이제스트와 LLM 입력에 실린다.

설계 규율 — 이 파이프라인에서 반복해 물렸던 실패를 막는다:

  1. **켜져 있는데 데이터가 없으면 실패한다.** 조용히 통과시키면, 가드가
     도는 줄 알고 안 도는 상태가 된다. 가드가 아예 없는 것보다 위험하다.
  2. **꺼진 것도 기록한다.** 매니페스트에 enabled/disabled 를 둘 다 남긴다.
     결과만 보고 "무엇이 돌았지?" 를 되묻는 상황이 없어야 한다.
  3. **gate_filter 는 단조여야 한다.** 이분탐색의 전제다. 설정으로 임의 식을
     넣을 수 있게 된 이상, 단조성은 검사해서 보장해야 한다.
  4. **섹터 적용성을 명시한다.** 부채비율·ROIC 는 금융업에서 의미가 없거나
     반대다(은행은 예금이 부채). 제외 섹터는 선택이 아니라 설계 항목이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from pipeline.screen.gate import FilterSpec

KINDS = ("hard_guard", "gate_filter", "soft_flag")
POLARITIES = ("risk", "positive", "neutral")
OPS = {">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
# 식에서 컬럼명을 뽑아내기 위한 식별자 패턴. 허용 함수 외에는 전부 컬럼으로 본다.
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_FUNCS = {"abs", "min", "max"}


class CheckConfigError(ValueError):
    """설정 자체가 성립하지 않는다. 실행 전에 잡는다."""


class CheckDataError(RuntimeError):
    """켜진 체크가 요구하는 데이터가 없다.

    MissingGuardError 와 같은 규율이다 — 조용히 건너뛰지 않는다.
    """


@dataclass(frozen=True)
class CheckSpec:
    id: str
    kind: str
    enabled: bool = False
    note: str = ""
    # hard_guard / soft_flag — 걸리면 True(=문제 있음)
    metric: str | None = None
    expr: str | None = None
    op: str | None = None
    threshold: float | None = None
    # gate_filter
    direction: str | None = None
    loose: float | None = None
    tight: float | None = None
    exclusive_zero: bool = False
    # 공통
    exclude_sectors: tuple[str, ...] = ()
    requires_source: str | None = None
    # 게재 판정용. risk 로 선언된 체크만 위험 점수에 들어간다.
    #   risk     — 재무 위험 신호
    #   positive — 가산 신호(고배당·자사주)
    #   neutral  — 중립이거나 데이터 품질 플래그(무배당, DART 수치 불일치)
    polarity: str = "neutral"
    # 위험 범주. low_roic 과 low_roe 처럼 같은 것을 두 번 말하는 플래그를
    # 개수로 세면 한 가지 약점이 둘로 계산된다. **범주 수**로 센다.
    risk_group: str | None = None
    # gate_filter 전용 커버리지 가드. 결측이 이 비율을 넘으면 실행을 거부한다.
    # higher_better 게이트에서 NaN 은 '탈락'으로 떨어지므로, 수집이 덜 된 채
    # 게이트를 켜면 미수집 종목이 **조용히 스크린에서 사라진다.**
    max_missing: float = 0.05

    @property
    def columns(self) -> set[str]:
        """이 체크가 읽는 컬럼."""
        if self.metric:
            return {self.metric}
        return {t for t in _IDENT.findall(self.expr or "") if t not in _ALLOWED_FUNCS}

    def as_filter_spec(self) -> FilterSpec:
        if self.kind != "gate_filter":
            raise CheckConfigError(f"{self.id}: gate_filter 가 아니다")
        return FilterSpec(metric=self.metric, direction=self.direction,
                          loose=self.loose, tight=self.tight,
                          exclusive_zero=self.exclusive_zero)


def load_checks(raw: list[dict] | None) -> list[CheckSpec]:
    """설정 → CheckSpec. 성립하지 않는 설정은 여기서 전부 잡는다."""
    specs: list[CheckSpec] = []
    seen: set[str] = set()
    for i, d in enumerate(raw or []):
        if not isinstance(d, dict) or "id" not in d:
            raise CheckConfigError(f"checks[{i}]: id 가 없다")
        cid = d["id"]
        if not _ID.match(str(cid)):
            raise CheckConfigError(f"{cid}: id 는 소문자·숫자·밑줄만 쓴다")
        if cid in seen:
            raise CheckConfigError(f"{cid}: id 가 중복이다")
        seen.add(cid)

        kind = d.get("kind")
        if kind not in KINDS:
            raise CheckConfigError(f"{cid}: kind 는 {KINDS} 중 하나여야 한다 (받은 값: {kind!r})")
        if bool(d.get("metric")) == bool(d.get("expr")):
            raise CheckConfigError(f"{cid}: metric 과 expr 중 정확히 하나만 지정한다")

        if kind == "gate_filter":
            if d.get("expr"):
                raise CheckConfigError(
                    f"{cid}: gate_filter 는 metric 만 쓴다 — 이분탐색이 단조성을 요구하는데 "
                    f"임의 식의 단조성은 보장할 수 없다")
            if d.get("direction") not in ("lower_better", "higher_better"):
                raise CheckConfigError(f"{cid}: gate_filter 는 direction 이 필요하다")
            if d.get("loose") is None or d.get("tight") is None:
                raise CheckConfigError(f"{cid}: gate_filter 는 loose 와 tight 가 필요하다")
        if d.get("polarity", "neutral") not in POLARITIES:
            raise CheckConfigError(
                f"{cid}: polarity 는 {POLARITIES} 중 하나여야 한다")
        if d.get("polarity") == "risk" and not d.get("risk_group"):
            raise CheckConfigError(
                f"{cid}: polarity=risk 면 risk_group 이 필요하다 — 같은 약점을 말하는 "
                f"플래그들이 개수로 중복 계산되는 것을 막기 위한 것이다")

        if kind == "gate_filter":
            pass
        else:
            if d.get("op") not in OPS:
                raise CheckConfigError(f"{cid}: op 는 {list(OPS)} 중 하나여야 한다")
            if d.get("threshold") is None:
                raise CheckConfigError(f"{cid}: threshold 가 필요하다")

        specs.append(CheckSpec(
            id=cid, kind=kind, enabled=bool(d.get("enabled", False)),
            note=d.get("note", ""), metric=d.get("metric"), expr=d.get("expr"),
            op=d.get("op"),
            threshold=None if d.get("threshold") is None else float(d["threshold"]),
            direction=d.get("direction"),
            loose=None if d.get("loose") is None else float(d["loose"]),
            tight=None if d.get("tight") is None else float(d["tight"]),
            exclusive_zero=bool(d.get("exclusive_zero", False)),
            exclude_sectors=tuple(d.get("exclude_sectors") or ()),
            requires_source=d.get("requires_source"),
            polarity=d.get("polarity", "neutral"),
            risk_group=d.get("risk_group"),
            max_missing=float(d.get("max_missing", 0.05)),
        ))
    return specs


def _series(df: pd.DataFrame, spec: CheckSpec) -> pd.Series:
    missing = spec.columns - set(df.columns)
    if missing:
        raise CheckDataError(
            f"체크 '{spec.id}' 가 켜져 있는데 필요한 컬럼이 없다: {sorted(missing)}. "
            f"끄거나(enabled: false) 해당 지표를 파생 단계에서 만들 것.")
    if spec.metric:
        return pd.to_numeric(df[spec.metric], errors="coerce")
    return pd.to_numeric(df.eval(spec.expr, engine="python"), errors="coerce")


def _hit(v: pd.Series, op: str, thr: float) -> pd.Series:
    """체크에 '걸렸는가'. 결측은 걸리지 않은 것으로 본다 —
    모르는 것을 위반으로 단정하면 데이터 결측이 곧 배제가 된다."""
    m = {">": v > thr, ">=": v >= thr, "<": v < thr, "<=": v <= thr}[op]
    return m.fillna(False)


@dataclass
class CheckReport:
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    hit_counts: dict[str, int] = field(default_factory=dict)
    exempt_counts: dict[str, int] = field(default_factory=dict)
    unevaluable: dict[str, int] = field(default_factory=dict)

    def manifest(self) -> dict:
        return {"enabled": self.enabled, "disabled": self.disabled,
                "hit_counts": self.hit_counts,
                "sector_exempt": {k: v for k, v in self.exempt_counts.items() if v},
                "unevaluable": {k: v for k, v in self.unevaluable.items() if v}}


def apply_checks(df: pd.DataFrame, specs: list[CheckSpec],
                 available_sources: set[str] | None = None,
                 ) -> tuple[pd.DataFrame, CheckReport]:
    """켜진 hard_guard / soft_flag 를 평가해 종목당 boolean 컬럼으로 붙인다.

    gate_filter 는 여기서 다루지 않는다 — 게이트 이분탐색이 소비한다.
    반환된 df 에는 체크 id 와 같은 이름의 컬럼이 생긴다(True = 걸림).
    """
    out = df.copy()
    rep = CheckReport()
    have = available_sources if available_sources is not None else set()

    for s in specs:
        if not s.enabled:
            rep.disabled.append(s.id)
            continue
        if s.requires_source and s.requires_source not in have:
            raise CheckDataError(
                f"체크 '{s.id}' 가 켜져 있는데 데이터 소스 '{s.requires_source}' 가 "
                f"수집되지 않았다. 수집하거나 체크를 끌 것.")
        rep.enabled.append(s.id)
        if s.kind == "gate_filter":
            continue

        v = _series(out, s)
        hit = _hit(v, s.op, s.threshold)
        if s.exclude_sectors:
            exempt = out["sector_code"].isin(s.exclude_sectors)
            rep.exempt_counts[s.id] = int(exempt.sum())
            hit &= ~exempt
        out[s.id] = hit
        rep.hit_counts[s.id] = int(hit.sum())
        rep.unevaluable[s.id] = int(v.isna().sum())

    return out, rep


def enabled_gate_filters(specs: list[CheckSpec]) -> list[FilterSpec]:
    return [s.as_filter_spec() for s in specs
            if s.enabled and s.kind == "gate_filter"]


def enabled_hard_guards(specs: list[CheckSpec]) -> list[str]:
    """하드 가드로 쓸 체크 id. apply_hard_guards 의 exclude_flags 에 합쳐진다."""
    return [s.id for s in specs if s.enabled and s.kind == "hard_guard"]


def enabled_soft_flags(specs: list[CheckSpec]) -> list[str]:
    return [s.id for s in specs if s.enabled and s.kind == "soft_flag"]


def assert_gate_coverage(df: pd.DataFrame, specs: list[CheckSpec]) -> None:
    """게이트 지표의 결측률이 허용치를 넘으면 실행을 거부한다.

    higher_better 게이트에서 NaN 은 마스크가 False 가 되어 그대로 탈락한다.
    수집이 덜 된 상태로 게이트를 켜면 '데이터를 못 받은 종목' 과
    '조건을 못 맞춘 종목' 이 구별되지 않은 채 결과에서 사라진다.
    """
    for s in specs:
        if not (s.enabled and s.kind == "gate_filter") or df.empty:
            continue
        missing = s.columns - set(df.columns)
        if missing:
            raise CheckDataError(
                f"게이트 체크 '{s.id}' 가 켜져 있는데 컬럼이 없다: {sorted(missing)}")
        frac = float(pd.to_numeric(df[s.metric], errors="coerce").isna().mean())
        if frac > s.max_missing:
            raise CheckDataError(
                f"게이트 체크 '{s.id}' 의 결측률이 {frac:.1%} 로 허용치 "
                f"{s.max_missing:.0%} 를 넘는다. 결측 종목은 게이트에서 조용히 "
                f"탈락하므로, 수집을 마치거나 max_missing 을 올려 명시적으로 감수할 것.")


def assert_monotone(df: pd.DataFrame, specs: list[CheckSpec], steps: int = 21) -> None:
    """gate_filter 가 t 에 대해 단조인지 확인한다.

    이분탐색은 'count(t) 가 비증가' 를 전제한다. 설정으로 임의 지표를 게이트에
    넣을 수 있게 된 이상, 전제가 깨지면 수렴이 조용히 엉뚱한 곳으로 간다.
    """
    for s in specs:
        if not (s.enabled and s.kind == "gate_filter"):
            continue
        f = s.as_filter_spec()
        prev = None
        for i in range(steps):
            t = i / (steps - 1)
            n = int(f.mask(df, t).sum())
            if prev is not None and n > prev:
                raise CheckConfigError(
                    f"체크 '{s.id}' 가 t 에 대해 단조가 아니다 "
                    f"(t={t:.2f} 에서 통과 {prev}→{n} 증가). "
                    f"loose 와 tight 방향이 direction 과 맞는지 확인할 것.")
            prev = n


def describe(specs: list[CheckSpec], df: pd.DataFrame | None = None) -> str:
    """`pipeline checks --list` 출력. 무엇이 켜져 있고 데이터가 있는지 한눈에."""
    lines = [f"{'':2} {'id':<24} {'kind':<12} {'조건':<28} {'데이터'}"]
    lines.append("-" * 88)
    for s in specs:
        mark = "ON " if s.enabled else "off"
        if s.kind == "gate_filter":
            cond = f"{s.direction} {s.loose}→{s.tight}"
        else:
            cond = f"{s.metric or s.expr} {s.op} {s.threshold}"
        if df is None:
            data = "-"
        else:
            missing = s.columns - set(df.columns)
            if missing:
                data = f"없음: {','.join(sorted(missing))}"
            else:
                try:
                    v = _series(df, s)
                    data = f"{int(v.notna().sum())}/{len(df)} 계산됨"
                except CheckDataError:
                    data = "평가 불가"
        lines.append(f"{mark} {s.id:<24} {s.kind:<12} {cond[:28]:<28} {data}")
    return "\n".join(lines)


def _hits(df: pd.DataFrame, s: CheckSpec) -> tuple[pd.Series, pd.Series, pd.Series]:
    v = _series(df, s)
    exempt = (df["sector_code"].isin(s.exclude_sectors)
              if s.exclude_sectors else pd.Series(False, index=df.index))
    return v, _hit(v, s.op, s.threshold) & ~exempt, exempt


def preview(df: pd.DataFrame, specs: list[CheckSpec],
            eligible: pd.DataFrame | None = None) -> str:
    """임계값을 바꾸기 전에 '몇 종목이 걸리는지' 를 먼저 본다.

    이게 없으면 절대 수치를 감으로 정하게 된다.

    두 모집단을 함께 보여주는 이유: 전체 유니버스에는 적자·자본잠식 종목이
    그대로 들어 있어 분포가 왜곡된다. 실제로 스크린이 다루는 것은
    하드가드를 통과한 쪽이고, 임계값은 거기서 판단해야 한다.
    """
    n_all = len(df)
    n_ok = len(eligible) if eligible is not None else 0
    head = f"{'id':<22} {'kind':<11} {'전체':>12}"
    if eligible is not None:
        head += f" {'가드통과':>12}"
    head += f" {'계산불가':>8}  가드통과 분포(p10/p50/p90)"
    lines = [head, "-" * (len(head) + 10)]

    for s in specs:
        mark = "" if s.enabled else " (off)"
        if s.kind == "gate_filter":
            lines.append(f"{s.id:<22} {s.kind:<11}  gate_filter 는 게이트가 소비{mark}")
            continue
        try:
            v, hit, exempt = _hits(df, s)
        except CheckDataError:
            miss = ",".join(sorted(s.columns - set(df.columns)))
            lines.append(f"{s.id:<22} {s.kind:<11}  데이터 없음: {miss}{mark}")
            continue
        cell_all = f"{int(hit.sum()):>5} ({hit.sum() / max(n_all, 1):>4.0%})"
        row = f"{s.id:<22} {s.kind:<11} {cell_all:>12}"
        pop = df
        if eligible is not None:
            try:
                _, hit_ok, _ = _hits(eligible, s)
                row += f" {f'{int(hit_ok.sum()):>5} ({hit_ok.sum() / max(n_ok, 1):>4.0%})':>12}"
                pop = eligible
            except CheckDataError:
                row += f" {'—':>12}"
        vp = _series(pop, s)
        q = (vp.dropna().quantile([0.1, 0.5, 0.9]).tolist()
             if vp.notna().any() else [float("nan")] * 3)
        row += (f" {int(v.isna().sum()):>8}  "
                f"{q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g}{mark}")
        lines.append(row)
    return "\n".join(lines)


# ── 게재 판정 (검증 축과 별개) ────────────────────────────────────────
def risk_groups(df: pd.DataFrame, specs: list[CheckSpec]) -> pd.Series:
    """종목별로 **몇 개 범주의** 재무 위험이 켜졌는가.

    tier 와 섞지 않는 이유: tier 는 '테마 배정이 검증됐는가' 한 축이다.
    거기에 재무 위험을 얹으면 tier A 정밀도가 분류 품질을 재지 못하게 되고,
    'A+B = 검증실패 0건' 항등식도 무너진다. 게재 여부는 별도 축으로 판정한다.

    개수가 아니라 범주 수를 세는 이유: low_roic 과 low_roe 는 같은 약점을
    두 번 말하는 것이라 개수로 세면 한 가지가 둘로 계산된다.
    """
    fired = pd.Series(0, index=df.index)
    if df.empty:
        return fired
    groups: dict[int, set[str]] = {i: set() for i in df.index}
    for s in specs:
        if not (s.enabled and s.polarity == "risk" and s.risk_group):
            continue
        if s.id not in df.columns:
            continue
        for i in df.index[df[s.id].fillna(False).astype(bool)]:
            groups[i].add(s.risk_group)
    return pd.Series({i: len(g) for i, g in groups.items()}).reindex(df.index).fillna(0).astype(int)


def risk_labels(row, specs: list[CheckSpec]) -> list[str]:
    """이 종목에서 켜진 위험 체크 id. 다이제스트 표기용."""
    return [s.id for s in specs
            if s.enabled and s.polarity == "risk"
            and s.id in row.index and bool(row[s.id])]
