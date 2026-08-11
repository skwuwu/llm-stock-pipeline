"""가격 소스 교차검증.

단일 소스로는 주가 오류를 잡을 수 없다 — 시총·PER·PBR 이 전부 같은 종가에서
나오므로 종가가 틀리면 셋이 함께 틀리고 서로를 검증하지 못한다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pipeline.ingest.price_sources import (OUT_COLUMNS, PriceSourceError,
                                           probe, reconcile)

AS_OF = date(2026, 8, 6)


def _frame(**px) -> pd.DataFrame:
    return pd.DataFrame({"ticker": list(px), "close": list(px.values()),
                         "adtv_20d": [1e9] * len(px),
                         "price_date": [AS_OF] * len(px)})


class _Stub:
    def __init__(self, name, frame=None, exc=None):
        self.name, self._f, self._e = name, frame, exc

    def fetch(self, tickers, as_of, refresh=False):
        if self._e:
            raise self._e
        return self._f


# ── 대조 ─────────────────────────────────────────────────────────────
def test_single_source_reports_that_it_could_not_verify():
    """소스가 하나면 disagree 는 전부 False 인데, 그건 '이상 없음'이 아니라
    '검증하지 못했음'이다. compared=0 으로 구별한다."""
    r = reconcile({"fdr": _frame(A=1000.0, B=2000.0)}, primary="fdr")
    assert not r.frame["price_source_disagree"].any()
    assert r.n_compared == 0 and r.compared_with == []
    assert r.manifest()["compared"] == 0


def test_agreeing_sources_produce_no_flag():
    r = reconcile({"fdr": _frame(A=1000.0), "kis": _frame(A=1005.0)}, primary="fdr")
    assert not r.frame["price_source_disagree"].iloc[0]
    assert r.n_compared == 1 and r.n_disagree == 0


def test_disagreement_is_flagged_but_value_is_kept():
    """어느 쪽이 맞는지 모를 때 임의로 하나를 고르면 틀린 값을 확신을 갖고 쓰게 된다.
    primary 를 그대로 두고 플래그만 단다 — 판정은 하드가드가 한다."""
    r = reconcile({"fdr": _frame(A=11510.0), "kis": _frame(A=153846.0)}, primary="fdr")
    row = r.frame.iloc[0]
    assert row["price_source_disagree"]
    assert row["close"] == 11510.0            # 값을 지우지 않는다
    assert row["price_alt_close"] == 153846.0  # 다른 쪽 값을 함께 남긴다
    assert row["price_source"] == "fdr"


def test_missing_in_one_source_is_not_a_disagreement():
    """한쪽에 없는 종목은 '불일치'가 아니라 '대조 불가'다."""
    r = reconcile({"fdr": _frame(A=1000.0, B=2000.0), "kis": _frame(A=1000.0)},
                  primary="fdr")
    assert r.n_compared == 1
    assert not r.frame.set_index("ticker").loc["B", "price_source_disagree"]


def test_unknown_primary_raises():
    with pytest.raises(PriceSourceError, match="primary"):
        reconcile({"kis": _frame(A=1.0)}, primary="fdr")


def test_tolerance_is_tight_enough_to_be_meaningful():
    """같은 거래일 종가는 원래 같아야 한다. 허용치가 넓으면 교차검증이 무의미해진다."""
    from pipeline.ingest.price_sources import PRICE_AGREEMENT_TOLERANCE
    assert PRICE_AGREEMENT_TOLERANCE <= 0.05
    r = reconcile({"a": _frame(X=100.0), "b": _frame(X=110.0)}, primary="a")
    assert r.frame["price_source_disagree"].iloc[0]


# ── 프로브 ───────────────────────────────────────────────────────────
def test_probe_rejects_source_that_returns_nothing():
    r = probe(_Stub("empty", _frame()), ["A"], AS_OF)
    assert not r.ok and "하나도" in r.detail


def test_probe_rejects_nonpositive_close():
    r = probe(_Stub("bad", _frame(A=0.0)), ["A"], AS_OF)
    assert not r.ok and "0 이하" in r.detail


def test_probe_rejects_missing_columns():
    r = probe(_Stub("thin", pd.DataFrame({"ticker": ["A"]})), ["A"], AS_OF)
    assert not r.ok and "컬럼 누락" in r.detail


def test_probe_surfaces_the_exception_instead_of_swallowing():
    r = probe(_Stub("boom", exc=PriceSourceError("자격증명 없음")), ["A"], AS_OF)
    assert not r.ok and "자격증명" in r.detail


def test_probe_passes_and_returns_sample():
    r = probe(_Stub("ok", _frame(A=1000.0)), ["A"], AS_OF)
    assert r.ok and r.sample is not None and len(r.sample) == 1


# ── 미검증 소스 게이트 ───────────────────────────────────────────────
def test_unverified_source_refuses_bulk_fetch(tmp_path):
    """추측으로 배선한 API 로 2,598종목을 돌리면 실패를 알아채는 시점이
    레이트 리밋을 소진한 뒤가 된다. 프로브 전에는 대량 수집을 거부한다."""
    from pipeline.ingest.kis import KisNotVerifiedError, KisSource
    src = KisSource(tmp_path / "kis")
    assert not src.verified
    with pytest.raises(KisNotVerifiedError, match="프로브"):
        src.fetch(["005930"], AS_OF)


def test_probe_can_bypass_its_own_gate(tmp_path, monkeypatch):
    """프로브가 검증 주체다 — 게이트가 프로브를 막으면 순환이 된다."""
    from pipeline.ingest.kis import KisSource
    src = KisSource(tmp_path / "kis")
    monkeypatch.setattr(src, "fetch", lambda t, a, refresh=False: _frame(A=1000.0))
    assert probe(src, ["A"], AS_OF).ok


def test_verification_persists_across_processes(tmp_path):
    """프로브와 수집은 다른 프로세스다. 메모리 플래그만으로는 게이트가 성립하지 않는다."""
    from pipeline.ingest.kis import KisSource
    root = tmp_path / "kis"
    KisSource(root).mark_verified(AS_OF, 4)
    assert KisSource(root).verified
    assert KisSource(root).verified_at["tickers_ok"] == 4


def test_kis_missing_credentials_is_reported_clearly(tmp_path, monkeypatch):
    """인증 실패를 종목별 except 로 삼키면 '자격증명 없음'이
    '종가를 하나도 못 받음'으로 둔갑해 원인이 감춰진다."""
    from pipeline.ingest.kis import KisSource
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_SECRET", raising=False)
    src = KisSource(tmp_path / "kis")
    src.app_key = src.app_secret = None
    src.verified = True                      # 게이트를 지나 인증까지 가게 한다
    with pytest.raises(PriceSourceError, match="자격증명"):
        src.fetch(["005930"], AS_OF)


# ── 실제 소스 ────────────────────────────────────────────────────────
def test_fdr_source_conforms_to_the_contract():
    """운영 소스가 계약(컬럼 4개)을 지키는지. 네트워크가 없으면 건너뛴다."""
    from pipeline.ingest.price_sources import FdrSource
    repo = Path(__file__).resolve().parents[1]
    r = probe(FdrSource(repo / "data/raw/prices"), ["005930"], AS_OF)
    if not r.ok:
        pytest.skip(f"FDR 사용 불가: {r.detail}")
    assert list(r.sample.columns) == OUT_COLUMNS
    assert r.sample["close"].iloc[0] > 0
