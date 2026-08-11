"""DART 주식종류 라벨 분류 — 8.9% 커버리지 구멍의 원인이었다.

`se` 는 열거형이 아니라 **자유 텍스트**다. 완전일치로 '보통주'만 보다가
230종목(삼성전기·LG전자·한미반도체·셀트리온·HD현대일렉트릭 …)의 주식수가
NULL 이 됐고, 시총이 없으니 하드가드가 '시총 미달'로 배제했다.
배제 사유가 데이터 결손처럼 보이지 않았다는 점이 이 버그의 핵심이다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from pipeline.ingest.prices import classify_share_class as cls

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/raw/dart/stockTotqySttus"


# ── 실측된 라벨 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("label", [
    "보통주", "보통주식", "기명식보통주", "의결권 있는 주식",
    "의결권 있는 주식(보통주)", "의결권있는주식\n(보통주)", "보통주\n(미래에셋증권)",
    "의결권이 있는 주식", "보통주(주1)",
    "보톧주", "보퉁주", "보통부",           # 실측된 오타
])
def test_common_labels(label):
    assert cls(label) == "common", label


@pytest.mark.parametrize("label", [
    "우선주", "우선주식", "종류주식", "종류주", "1종 종류주식", "제2우선주",
    "무의결권부우선주", "전환우선주", "상환전환우선주", "배당우선전환주식",
    "의결권 없는 주식", "의결권없는주식", "무의결권\n 배당우선주",
    "신형우선주", "우선주A", "1우선주\n(미래에셋증권우)",
])
def test_other_class_labels(label):
    assert cls(label) == "other", label


@pytest.mark.parametrize("label", ["합계", "비고", "기타", "유통주식", "주식수", "", None])
def test_aggregate_rows_are_skipped(label):
    """None 과 'other' 는 다르다 — 전자는 '이 행을 무시하라'다.
    합계 행을 우선주로 세면 시총이 두 배가 된다."""
    assert cls(label) is None, label


def test_preferred_marks_win_over_voting_marks():
    """'의결권 있는 주식(우선주)' 는 우선주다. 순서가 곧 규칙이라
    우선주 계열을 먼저 걸러야 혼합 표기가 보통주로 새지 않는다."""
    assert cls("의결권 있는 주식(우선주)") == "other"
    assert cls("의결권 없는 주식(우선주)") == "other"


def test_newlines_and_spacing_normalized():
    assert cls("의결권 있는\n주식") == cls("의결권있는 주식") == "common"
    assert cls("의결권\n없는 주식") == "other"


# ── 원천 전수 ────────────────────────────────────────────────────────
def _payloads():
    for f in RAW.glob("*.json"):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            continue
        if p.get("status") == "000":
            yield f.name, p


def test_almost_every_filing_yields_a_common_row():
    """실측 기준선: 2,704개 중 보통주를 못 찾는 것은 5건뿐이고,
    그 5건은 DART 가 합계·비고 행만 준 경우다(파서로 고칠 수 없다).

    이 수가 늘면 새 라벨 표기가 등장했다는 뜻이다.
    """
    if not RAW.exists():
        pytest.skip("원천 없음")
    total = miss = 0
    for name, p in _payloads():
        total += 1
        if not any(cls(r.get("se")) == "common" for r in p.get("list", [])):
            miss += 1
    assert total > 2000, f"원천이 너무 적다({total}) — 경로를 확인할 것"
    assert miss <= 10, f"보통주 행을 못 찾는 파일 {miss}/{total} — 새 라벨 표기 의심"


def test_no_filing_classifies_the_total_row_as_a_share_class():
    """합계 행이 섞이면 발행주식수가 두 배가 되고 시총·PBR 이 전부 틀린다."""
    if not RAW.exists():
        pytest.skip("원천 없음")
    for name, p in _payloads():
        for r in p.get("list", []):
            se = " ".join((r.get("se") or "").split())
            if "합계" in se:
                assert cls(se) is None, f"{name}: 합계 행이 {cls(se)} 로 분류됐다"


def test_unclassified_labels_stay_rare():
    """미분류가 늘면 조용히 종목이 빠진다. 상한을 걸어 눈에 보이게 한다."""
    if not RAW.exists():
        pytest.skip("원천 없음")
    unk = Counter()
    for name, p in _payloads():
        for r in p.get("list", []):
            se = " ".join((r.get("se") or "").split())
            if se and "합계" not in se and se != "비고" and cls(se) is None:
                unk[se] += 1
    assert sum(unk.values()) <= 40, f"미분류 라벨이 늘었다: {unk.most_common(10)}"


# ── 회귀 방지: 커버리지 ──────────────────────────────────────────────
def test_market_cap_coverage_regression():
    """시총 결측률. 파서 수정 전 8.9%(230/2598) 였다.

    이 값이 다시 올라가면 라벨 표기가 바뀌었거나 수집이 깨진 것이다.
    """
    import pandas as pd
    p = REPO / "data/derived/metrics_2026-08-06.parquet"
    if not p.exists():
        pytest.skip("파생지표 없음")
    m = pd.read_parquet(p)
    rate = float(m["market_cap_used"].isna().mean())
    assert rate < 0.05, (
        f"시총 결측 {rate:.1%} — 주식수 파싱이 다시 깨졌을 수 있다. "
        f"classify_share_class 와 원천 se 라벨을 대조할 것")
