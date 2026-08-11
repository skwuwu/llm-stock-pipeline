"""골든셋 대비 테마 오분류율 측정.

측정 대상은 **core(주력) 테마**로 한정한다. adjacent/peripheral 은 사람끼리도
판정이 갈려 라벨의 신뢰도가 떨어지고, tier A 후보는 어차피 core 뿐이다.

가장 중요한 지표는 절대 정밀도가 아니라 **검증 전후 차이**다.
검증 레이어가 정밀도를 올리지 못하면 그 레이어는 장식이다.

⚠ 모든 행이 같은 무게를 갖지 않는다 — REPRODUCIBILITY 참조.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
GOLDEN = REPO / "tests" / "golden" / "kr_core_themes_v1.jsonl"


# ── 재현성 실측값 ────────────────────────────────────────────────────
# scripts/determinism_probe.py 로 캐시를 우회해 같은 pack 을 두 번 태깅한 결과.
#
# 왜 여기 적어두는가: 지표를 읽는 사람이 "95.7% 가 98.0% 보다 나빠졌다"고 읽기
# 전에, 그 차이가 잴 수 있는 크기인지부터 알아야 한다. 실제로 택소노미를
# 확장한 직후 A+B 재현율이 80.3% → 73.8% 로 내려갔고 하마터면 택소노미 탓으로
# 결론 낼 뻔했다. 원인은 실행 간 흔들림이었다.
#
# core 가 판정보다 안정적인 이유: core 는 LLM 이 고르는 **이산 선택**이다.
# 판정은 세그먼트 이름 매칭(LLM 출력)을 거쳐 실측 비중을 찾으므로 한 단계가
# 더 끼고, 그만큼 더 흔들린다.
#
# Sonnet 5 는 temperature 를 받지 않아(400 deprecated) 승급된 호출은
# 온도로 고정할 수 없다 — cascade.TEMPERATURE_SUPPORTED 참조. 승급률이 85% 라
# **대부분의 호출이 온도 고정 밖에 있다.** 그래서 흔들림은 없앨 수 없고,
# 크기를 알고 그보다 작은 차이를 개선이라 부르지 않는 것이 최선이다.
# ⚠ **한 번 재고 100% 라고 말하면 안 된다.** 첫 표본(n=24)에서 core 일치가
# 24/24 로 나와 '완전 재현'이라 적었는데, 리팩터 후 같은 크기로 다시 재니
# 22/24 였다. 24/24 는 운이었다 — 두 표본을 합쳐야 96%(46/48)다.
# 그래서 이 값은 표본을 누적해 기록하고, 단일 실행의 수치를 쓰지 않는다.
_SAMPLES = [
    # (측정일, 종목수, core 일치, 배정수, 판정 일치, 비고)
    ("2026-08-08", 24, 24, 38, 32, "등급 체계(A/B/C), 온도 기본값"),
    ("2026-08-08", 24, 22, 37, 29, "verified/rejected, Haiku 온도 0"),
]
_CORE_N = sum(s[1] for s in _SAMPLES)
_CORE_OK = sum(s[2] for s in _SAMPLES)
_ASG_N = sum(s[3] for s in _SAMPLES)
_ASG_OK = sum(s[4] for s in _SAMPLES)

REPRODUCIBILITY = {
    "measured_at": _SAMPLES[-1][0],
    "screen": "deep_value",
    "samples": len(_SAMPLES),
    "n_stocks": _CORE_N,
    "n_assignments": _ASG_N,
    # 오분류율의 계산 단위. 흔들리는 쪽은 대부분 **재현율**이다 — 실측 2건
    # 모두 '한 실행이 정답 core 를 놓치고 다른 실행은 찾은' 형태였다
    # (아이에스동서 recycling_environment, 삼호개발 construction_materials).
    "core_agreement": round(_CORE_OK / _CORE_N, 3),          # 46/48 = 0.958
    # (status, role, share_evidence) 삼중조. 등급을 없앤 뒤 리포트에 실리는 전부.
    "verdict_agreement": round(_ASG_OK / _ASG_N, 3),         # 61/75 = 0.813
    "full_agreement": 0.38,      # role·confidence 소수점까지. 참고용 하한
    # **지표 차이를 읽을 때의 잡음 폭.** core 일치 96% 는 62종목 코호트에서
    # 배정 2~3건이 실행마다 오간다는 뜻이고, 재현율로는 대략 ±4%p 다.
    # 이보다 작은 변화를 개선이라 부르면 안 된다.
    "recall_noise_pp": 4,
    "tool": "scripts/determinism_probe.py",
    "note": "표본을 합쳐서 본다. 단일 실행의 수치는 과대·과소 양쪽으로 튄다.",
}


@dataclass
class GoldLabel:
    ticker: str
    name: str
    core: set[str]
    evidence_defect: bool = False
    note: str = ""
    # certain  : 규칙을 적용하면 다른 결론이 나오기 어렵다
    # contested: 사람에 따라 갈릴 수 있다(테마 경계·비중 해석). 이유를 note 에 남긴다
    #
    # 오탐 분모가 한 자릿수일 때 라벨 1건이 지표를 뒤집는다. 흔들릴 수 있는 라벨을
    # 표시해 두고 certain-only 지표를 함께 보면, 숫자가 라벨 판단에 얼마나 기대고
    # 있는지가 드러난다.
    confidence: str = "certain"


def load_golden(path: Path | None = None) -> dict[str, GoldLabel]:
    out: dict[str, GoldLabel] = {}
    for line in (path or GOLDEN).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "_meta" in d:
            continue
        conf = d.get("confidence", "certain")
        if conf not in ("certain", "contested"):
            raise ValueError(f"{d['ticker']}: confidence 는 certain|contested (받은 값 {conf!r})")
        out[d["ticker"]] = GoldLabel(d["ticker"], d.get("name", ""),
                                     set(d.get("core") or []),
                                     bool(d.get("evidence_defect")),
                                     d.get("note", ""), conf)
    return out


@dataclass
class Scores:
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    stocks: int = 0
    correct_abstain: int = 0
    gold_empty: int = 0
    fp_by_theme: dict[str, int] = field(default_factory=dict)
    fn_by_theme: dict[str, int] = field(default_factory=dict)
    # 이 행을 재실행해도 같은 값이 나오는가. False 면 개선 지표로 쓸 수 없다 —
    # 두 실행의 차이가 개선인지 흔들림인지 구분되지 않기 때문이다.
    reproducible: bool = False

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def misclassification_rate(self) -> float | None:
        """예측한 core 중 골든셋에 없는 비율. 오탐 지표."""
        p = self.precision
        return None if p is None else 1 - p

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return None if not p or not r else 2 * p * r / (p + r)

    def as_dict(self) -> dict:
        def pct(x):
            return None if x is None else round(x, 4)
        return {
            "label": self.label, "stocks": self.stocks,
            "reproducible": self.reproducible,
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": pct(self.precision), "recall": pct(self.recall),
            "f1": pct(self.f1),
            "misclassification_rate": pct(self.misclassification_rate),
            "correct_abstain": f"{self.correct_abstain}/{self.gold_empty}",
            "top_false_positives": dict(sorted(self.fp_by_theme.items(),
                                               key=lambda x: -x[1])[:5]),
            "top_false_negatives": dict(sorted(self.fn_by_theme.items(),
                                               key=lambda x: -x[1])[:5]),
        }


def score(predicted: dict[str, set[str]], gold: dict[str, GoldLabel],
          label: str, skip_evidence_defect: bool = False,
          certain_only: bool = False, reproducible: bool = False) -> Scores:
    s = Scores(label=label, reproducible=reproducible)
    for ticker, g in gold.items():
        if skip_evidence_defect and g.evidence_defect:
            continue
        if certain_only and g.confidence != "certain":
            continue
        pred = predicted.get(ticker, set())
        s.stocks += 1
        if not g.core:
            s.gold_empty += 1
            if not pred:
                s.correct_abstain += 1
        for t in pred - g.core:
            s.fp += 1
            s.fp_by_theme[t] = s.fp_by_theme.get(t, 0) + 1
        for t in g.core & pred:
            s.tp += 1
        for t in g.core - pred:
            s.fn += 1
            s.fn_by_theme[t] = s.fn_by_theme.get(t, 0) + 1
    return s


# ── 예측 추출 ────────────────────────────────────────────────────────
def cores_from_tags(tags: list[dict]) -> dict[str, set[str]]:
    """검증 **전**: LLM 이 core 라고 말한 것 그대로."""
    out: dict[str, set[str]] = {}
    for t in tags:
        out[t["ticker"]] = {a["theme_id"] for a in (t.get("assignments") or [])
                            if a.get("role") == "core"}
    return out


def cores_from_verdicts(verdicts, share_evidence: set[str] | None = None,
                        clean_only: bool = False) -> dict[str, set[str]]:
    """검증 **후**: 폐기와 역할 강등을 반영한 core.

    등급(A/B/C)은 더 이상 없다 — layers 모듈 docstring 참조. 남은 축은 둘이다.

    share_evidence  비중 근거로 좁힌다. 예: {"confirmed"} 면 실측 세그먼트
                    수치가 하한을 넘은 것만. 좁힐수록 정밀도는 오르고
                    재현율은 떨어지는데, 그건 분류 실패가 아니라
                    **세그먼트를 공시하지 않는 회사가 많다는 사실**이다.
    clean_only      플래그가 하나도 없는 배정만. 플래그에 신호가 있는지
                    재는 대조군이다.
    """
    out: dict[str, set[str]] = {}
    for v in verdicts:
        g = (lambda k: v[k]) if isinstance(v, dict) else (lambda k: getattr(v, k))
        tk = g("ticker")
        out.setdefault(tk, set())
        if g("status") != "verified" or g("role") != "core":
            continue
        if share_evidence is not None and g("share_evidence") not in share_evidence:
            continue
        if clean_only:
            fl = g("flags")
            fl = [f for f in fl.split("|") if f] if isinstance(fl, str) else (fl or [])
            if fl:
                continue
        out[tk].add(g("theme_id"))
    return out
