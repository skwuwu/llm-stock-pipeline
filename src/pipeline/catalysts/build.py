"""공시 → 촉매. **LLM 을 쓰지 않는다.**

카탈로그(configs/catalysts/catalyst_v1.yaml)가 report_nm 패턴으로 kind 를
정하고, extract 가 숫자를 뽑고, 여기서 신뢰도 6체크를 매긴다. 전부 데이터
조회라 재실행해도 같은 값이 나온다 — 직전에 등급 체계를 없앤 이유가
'LLM 이 매긴 소수점이 순위를 정하면 순위가 매번 바뀐다' 였고, 여기서는
그 문제가 구조적으로 발생하지 않는다.

LLM 은 나중에 **인용문과 40자 논거**를 붙일 때만 개입한다. 그때도 kind /
magnitude / expires_at 은 건드리지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
CATALOG = REPO / "configs" / "catalysts" / "catalyst_v1.yaml"

# 공시 본문을 파싱하지 않고 **우리 파생지표를 크기로 쓰는** 촉매.
# C3(실적 서프라이즈)가 그렇다: 잠정실적 공시 3,270건을 전부 받아 파싱하는
# 대신, 공시는 날짜 앵커로만 쓰고 크기는 op_delta_q(증가액)에서 가져온다.
# **증가율이 아니라 증가액이다** — 비율은 기저가 작으면 폭주한다.
SELF_COMPUTED = {"C3": "op_delta_q"}


@dataclass
class CatalystSpec:
    id: str
    name: str
    polarity: str
    enabled: bool
    pblntf_ty: list[str]
    pattern: re.Pattern
    mag_denominator: str | None
    mag_min: float | None
    expires_days: int | None


def load_catalog(path: Path | None = None) -> tuple[list[CatalystSpec], dict]:
    cfg = yaml.safe_load((path or CATALOG).read_text(encoding="utf-8"))
    specs = []
    for c in cfg["catalysts"]:
        pats = c.get("patterns") or []
        mag = c.get("magnitude") or {}
        specs.append(CatalystSpec(
            id=c["id"], name=c["name"], polarity=c["polarity"],
            enabled=bool(c.get("enabled")), pblntf_ty=list(c["pblntf_ty"]),
            pattern=re.compile("|".join(pats)) if pats else re.compile(r"(?!)"),
            mag_denominator=mag.get("denominator"),
            mag_min=mag.get("min"),
            expires_days=c.get("expires_days"),
        ))
    return specs, cfg


@dataclass
class Catalyst:
    ticker: str
    kind: str
    name: str
    polarity: str
    rcept_no: str
    occurred_at: date
    expires_at: date | None
    magnitude: float | None
    magnitude_basis: str | None
    report_key: str
    # 신뢰도 6체크. **전부 데이터 조회다.** LLM 자기신고가 들어갈 자리가 없다.
    checks: dict = field(default_factory=dict)

    @property
    def confidence(self) -> int:
        return sum(1 for v in self.checks.values() if v)

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("checks")
        d["confidence"] = self.confidence
        d.update({f"chk_{k}": v for k, v in self.checks.items()})
        return d


def classify(disclosures: pd.DataFrame, specs: list[CatalystSpec],
             enabled_only: bool = True) -> pd.DataFrame:
    """공시 → (kind 가 붙은) 촉매 후보. 패턴 매칭뿐, 숫자는 아직 없다."""
    out = []
    for s in specs:
        if enabled_only and not s.enabled:
            continue
        m = (disclosures["pblntf_ty"].isin(s.pblntf_ty)
             & disclosures["report_key"].str.contains(s.pattern, regex=True)
             & disclosures["ticker"].notna())
        if m.any():
            out.append(disclosures[m].assign(kind=s.id))
    if not out:
        return disclosures.head(0).assign(kind=pd.Series(dtype=str))
    return pd.concat(out, ignore_index=True)


def amendment_index(disclosures: pd.DataFrame) -> set[tuple[str, str]]:
    """(ticker, report_key) → 정정공시가 존재하는 집합.

    DART 는 정정 대상 rcept_no 를 목록 API 로 주지 않는다. 그래서
    **같은 종목의 같은 보고서명에 정정본이 있으면 원본을 의심**하는 것이
    목록만으로 할 수 있는 최선이다. 과탐 쪽으로 치우친 근사다 —
    같은 종류의 다른 건을 정정한 경우도 걸린다. 그 편향을 알고 쓴다.
    """
    a = disclosures[disclosures["is_amendment"] & disclosures["ticker"].notna()]
    return set(zip(a["ticker"], a["report_key"]))


def build(cands: pd.DataFrame, specs: dict[str, CatalystSpec],
          mags: dict[str, tuple[float | None, str | None, date | None]],
          metrics: pd.DataFrame, as_of: date,
          amended: set[tuple[str, str]]) -> list[Catalyst]:
    """후보 + 크기 → 촉매. 신뢰도 6체크를 여기서 매긴다."""
    from pipeline.catalysts.extract import normalize, resolve_expiry

    # DuckDB 는 DATE 를 pandas Timestamp 로 돌려준다. date 와 비교하면
    # TypeError 라, 경계에서 한 번만 맞춰 둔다 — 아래 로직 전체가 date 로 논다.
    cands = cands.copy()
    cands["rcept_dt"] = pd.to_datetime(cands["rcept_dt"]).dt.date
    if hasattr(as_of, "date") and not isinstance(as_of, date):
        as_of = as_of.date()

    m = metrics.set_index("ticker") if "ticker" in metrics.columns else metrics
    # 반대 극성 촉매의 최신 발생일. no_reversal 체크에 쓴다.
    latest_opposite: dict[tuple[str, str], date] = {}
    for r in cands.itertuples():
        pol = specs[r.kind].polarity
        k = (r.ticker, "negative" if pol == "positive" else "positive")
        d = r.rcept_dt
        if k not in latest_opposite or d > latest_opposite[k]:
            latest_opposite[k] = d

    out: list[Catalyst] = []
    for r in cands.itertuples():
        sp = specs[r.kind]
        raw, basis, doc_exp = mags.get(r.rcept_no, (None, None, None))
        if raw is None and r.kind in SELF_COMPUTED:
            # **자체 계산 경로.** 잠정실적 공시를 파싱하지 않고 우리 지표를 쓴다.
            # 같은 수치를 두 경로로 만들면 어긋날 때 어느 쪽이 맞는지 판정할
            # 근거가 없다 — 하나로 고정하는 편이 정직하다. 공시는 '언제
            # 알려졌나'(occurred_at)와 '실제로 냈나'(rcept_no)만 제공한다.
            col = SELF_COMPUTED[r.kind]
            if r.ticker in m.index:
                v = m.loc[r.ticker].get(col)
                if pd.notna(v):
                    raw, basis = float(v), f"metrics:{col}"
        denom = None
        if sp.mag_denominator and r.ticker in m.index:
            denom = m.loc[r.ticker].get(sp.mag_denominator)
        # denominator 가 null 인 촉매(C2·C3)는 raw 가 이미 비율이다
        mag = raw if sp.mag_denominator is None else normalize(raw, denom)
        exp = resolve_expiry(r.rcept_dt, sp.expires_days, doc_exp)

        opp = latest_opposite.get((r.ticker, sp.polarity))
        checks = {
            # grounded 는 인용문이 붙은 뒤에 채운다. 지금은 rcept_no 존재로 대체.
            "grounded": bool(r.rcept_no),
            "not_amended": (r.ticker, r.report_key) not in amended,
            "not_expired": exp is None or exp >= as_of,
            "material": (mag is not None and sp.mag_min is not None
                         and mag >= sp.mag_min) or sp.mag_min is None,
            "no_reversal": opp is None or opp <= r.rcept_dt,
            "numbers_agree": _numbers_agree(r.kind, r.ticker, m),
        }
        out.append(Catalyst(
            ticker=r.ticker, kind=r.kind, name=sp.name, polarity=sp.polarity,
            rcept_no=r.rcept_no, occurred_at=r.rcept_dt, expires_at=exp,
            magnitude=mag, magnitude_basis=basis, report_key=r.report_key,
            checks=checks))
    return out


def _numbers_agree(kind: str, ticker: str, m: pd.DataFrame) -> bool:
    """우리 파생지표와 방향이 맞는가.

    magnitude 를 자체 지표에서 가져오는 촉매(C3)는 **다른 기간**으로 확인한다.
    같은 지표로 다시 보면 동어반복이라 체크가 아무것도 걸러내지 못한다.
    """
    if ticker not in m.index:
        return False
    row = m.loc[ticker]
    if kind == "C3":
        # magnitude 를 우리 분기 지표에서 가져오므로, 같은 지표로 다시 확인하면
        # 동어반복이다. **다른 기간**으로 본다 — 분기 서프라이즈가 연간 추세와
        # 어긋나면 일회성일 수 있다. 결측은 '확인 불가'라 실패로 세지 않는다
        # (미공시를 벌하지 않는 이 파이프라인의 일관된 규칙).
        g = row.get("op_growth_fy")
        return bool(pd.isna(g) or float(g) > 0)
    if kind == "C1":
        # 자사주를 사려면 현금이 있어야 한다. FCF 가 음수인데 자사주 취득은
        # 차입으로 사는 것이라 지속되지 않는다.
        f = row.get("fcf")
        return bool(pd.isna(f) or float(f) > 0)
    return True
