"""M1: 섹터 매핑 + 종목 상태 가드. 네트워크 없이 돈다.

kind_industries.txt 는 KIND 실제 업종 목록(159종) 스냅샷이다.
매핑 설정을 고치다 커버리지가 깨지면 여기서 잡힌다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipeline.derive.metrics import add_quality_flags
from pipeline.normalize.sector import annotate, coverage_report, looks_like_holding, map_industry
from pipeline.store.pit import PitStore

FIX = Path(__file__).parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


def industries() -> list[tuple[int, str]]:
    out = []
    for line in (FIX / "kind_industries.txt").read_text(encoding="utf-8").splitlines():
        n, name = line.split("\t", 1)
        out.append((int(n), name))
    return out


# ── 섹터 매핑 ───────────────────────────────────────────────────────
def test_every_known_industry_maps():
    """159개 업종 전부 매핑돼야 한다.

    미매핑 종목은 V2 섹터 정합성 검증이 아무 일도 하지 않으므로,
    '검증이 돌고 있다'는 착각을 만든다.
    """
    unmapped = [(n, name) for n, name in industries() if map_industry(name) is None]
    assert not unmapped, f"미매핑 업종 {len(unmapped)}종: {unmapped[:5]}"


def test_mapping_codes_exist_in_sector_map():
    sec = yaml.safe_load((REPO / "configs/sectors/sector_map.yaml").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((REPO / "configs/sectors/kind_industry_map.yaml").read_text(encoding="utf-8"))
    unknown = {r["code"] for r in cfg["rules"]} - set(sec["codes"])
    assert not unknown, f"sector_map 미등록 코드: {unknown}"


@pytest.mark.parametrize("industry, expected", [
    ("전동기, 발전기 및 전기 변환 · 공급 · 제어 장치 제조업", "ELEC_EQUIP"),  # AI 전력기기 모집단
    ("절연선 및 케이블 제조업", "ELEC_EQUIP"),
    ("반도체 제조업", "SEMI"),
    ("선박 및 보트 건조업", "SHIPBUILD"),
    ("항공기,우주선 및 부품 제조업", "AEROSPACE_DEFENSE"),
    ("무기 및 총포탄 제조업", "AEROSPACE_DEFENSE"),
    ("은행 및 저축기관", "BANK"),
    ("금융 지원 서비스업", "SECURITIES"),
    ("기타 금융업", "OTHER_FIN"),
    ("재 보험업", "INSURANCE"),
    ("소프트웨어 개발 및 공급업", "IT_SERVICES"),
    ("일차전지 및 이차전지 제조업", "ELECTRONIC_COMP"),
    ("비알코올음료 및 얼음 제조업", "FOOD_BEV"),
    ("일반 교습 학원", "EDUCATION"),
])
def test_representative_mappings(industry, expected):
    assert map_industry(industry) == expected


def test_finance_not_swallowed_by_generic_rules():
    """'기타 금융업'이 '기타 ~ 서비스업' 폴백에 먹히면 안 된다(순서 의존)."""
    assert map_industry("기타 금융업") == "OTHER_FIN"
    assert map_industry("그외 기타 전문, 과학 및 기술 서비스업") == "BUSINESS_SVC"


def test_annotate_and_coverage():
    df = pd.DataFrame({
        "ticker": ["000001", "000002", "000003", "000004"],
        "name": ["가나홀딩스", "다라스팩1호", "마바리츠", "사아전자"],
        "industry": ["기타 금융업", "금융 지원 서비스업", "부동산 임대 및 공급업",
                     "전자부품 제조업"],
        "fiscal_month": [12, 12, 12, 3],
    })
    out = annotate(df)
    assert list(out["sector_code"]) == ["OTHER_FIN", "SECURITIES", "REAL_ESTATE",
                                        "ELECTRONIC_COMP"]
    assert list(out["is_holding"]) == [True, False, False, False]
    assert list(out["is_spac"]) == [False, True, False, False]
    assert list(out["is_reit"]) == [False, False, True, False]
    assert list(out["is_financial"]) == [True, True, False, False]
    assert coverage_report(out)["coverage"] == 1.0


def test_holding_heuristic_is_name_based():
    assert looks_like_holding("LG홀딩스")
    assert looks_like_holding("한국지주")
    assert not looks_like_holding("삼성전자")


# ── 상태 가드 (PIT) ─────────────────────────────────────────────────
@pytest.fixture
def store(tmp_path):
    s = PitStore(tmp_path / "p.duckdb")
    s.upsert_status(pd.DataFrame([
        {"ticker": "000001", "status": "admin_issue", "effective_from": date(2024, 3, 1),
         "reason": "감사의견 거절", "source": "kind", "snapshot_date": date(2026, 8, 7)},
        {"ticker": "000001", "status": "audit_opinion_bad_admin",
         "effective_from": date(2024, 3, 1), "reason": "감사의견 거절",
         "source": "kind", "snapshot_date": date(2026, 8, 7)},
        {"ticker": "000002", "status": "admin_issue", "effective_from": date(2026, 5, 1),
         "reason": "시가총액 미달", "source": "kind", "snapshot_date": date(2026, 8, 7)},
        {"ticker": "000003", "status": "admin_issue", "effective_from": None,
         "reason": "지정일 불명", "source": "kind", "snapshot_date": date(2026, 8, 7)},
    ]))
    yield s
    s.close()


def test_status_respects_designation_date(store):
    """지정일 이후의 as_of 에서만 잡혀야 한다. 미래 지정이 과거로 새면 안 된다."""
    assert set(store.status_asof(date(2024, 1, 1))["ticker"]) == set()
    assert set(store.status_asof(date(2025, 1, 1))["ticker"]) == {"000001"}
    assert set(store.status_asof(date(2026, 8, 7))["ticker"]) == {"000001", "000002"}


def test_status_without_effective_date_is_excluded(store):
    """지정일이 없으면 언제부터인지 모르므로 어떤 as_of 에도 적용하지 않는다."""
    assert "000003" not in set(store.status_asof(date(2026, 8, 7))["ticker"])


def _metrics_stub() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["000001", "000002"],
        "equity_controlling": [1e12, 1e12], "equity_total": [1e12, 1e12],
        "net_income_ttm": [1e11, 1e11], "operating_income_ttm": [1.2e11, 1.2e11],
        "per_op": [10.0, 10.0], "per": [12.0, 12.0], "fcf": [1e11, 1e11],
        "pbr": [1.2, 1.2], "market_cap_used": [1.2e12, 1.2e12],
        "cfo_ttm": [1.5e11, 1.5e11], "ni_ttm_reason": ["annual", "annual"],
        "fiscal_month": [12, 12], "is_financial": [False, False],
        "is_holding": [False, False], "sector_code": ["SEMI", None],
        "equity_basis": ["CFS", "CFS"],
    })


def test_status_flags_applied(store):
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    out = add_quality_flags(_metrics_stub(), facts, date(2026, 8, 7),
                            status=store.status_asof(date(2026, 8, 7)))
    assert list(out["admin_issue"]) == [True, True]
    assert list(out["audit_opinion_bad_admin"]) == [True, False]
    assert not out["status_source_missing"].any()


def test_missing_status_source_is_visible():
    """status 를 안 넘기면 플래그는 전부 False 지만, 그 사실이 드러나야 한다."""
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    out = add_quality_flags(_metrics_stub(), facts, date(2026, 8, 7), status=None)
    assert not out["admin_issue"].any()
    assert out["status_source_missing"].all()


def test_unmapped_industry_flagged():
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    out = add_quality_flags(_metrics_stub(), facts, date(2026, 8, 7))
    assert list(out["unmapped_industry"]) == [False, True]


# ── 설정 정합성 ─────────────────────────────────────────────────────
def test_screen_config_guards_are_producible():
    """설정이 요구하는 하드 가드를 metrics 가 실제로 만들어내는지."""
    cfg = yaml.safe_load((REPO / "configs/screen/kr_deep_value.yaml").read_text(encoding="utf-8"))
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    produced = set(add_quality_flags(_metrics_stub(), facts, date(2026, 8, 7)).columns)
    produced |= {"is_spac", "is_reit", "is_preferred"}   # master 에서 병합되는 것
    missing = set(cfg["universe"]["exclude_flags"]) - produced
    assert not missing, f"설정이 요구하지만 생성되지 않는 가드: {missing}"


# ── 물리적 불가능 값 가드 ───────────────────────────────────────────
def test_implausible_market_cap_blocked():
    """실측 사례: DART 주식총수가 30.6조 주로 보고돼 시총이 1.4해원이 됐다.
    하한(min_market_cap)만 두면 '너무 큰' 오류는 그대로 통과한다."""
    df = _metrics_stub()
    df.loc[0, "market_cap_used"] = 1.4e18
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    out = add_quality_flags(df, facts, date(2026, 8, 7))
    assert list(out["market_cap_implausible"]) == [True, False]


def test_implausible_pbr_blocked():
    """외부 소스 없이 시총·자본 정합성을 교차검증한다."""
    df = _metrics_stub()
    df.loc[0, "pbr"] = 5000.0        # 자본 대비 시총이 물리적으로 불가능
    df.loc[1, "pbr"] = 3.0           # 고평가지만 가능한 값 — 죽이면 안 된다
    facts = pd.DataFrame({"ticker": ["000001", "000002"],
                          "reported_at": [date(2026, 6, 1)] * 2})
    out = add_quality_flags(df, facts, date(2026, 8, 7))
    assert list(out["pbr_implausible"]) == [True, False]
