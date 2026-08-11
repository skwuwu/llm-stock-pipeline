"""스캔 주기와 리밸런스 주기 분리.

히스테리시스는 '직전 **리밸런스**' 를 기준으로 해야 한다. '직전 실행' 을 쓰면
일별로 돌릴 때 기준이 매일 어제로 밀려, 임계값 근처 종목의 잦은 교체를 막으려던
장치가 정반대로 작동한다 — 바스켓이 서서히 표류한다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pipeline.screen import basket as bk

D1, D2 = date(2026, 8, 6), date(2026, 8, 20)


def test_missing_file_is_empty_basket_not_an_error():
    """최초 실행에는 바스켓이 없다. 그때 히스테리시스는 적용되지 않는다."""
    b = bk.load(Path("does_not_exist.json"))
    assert not b.exists and b.members == set()


def test_roundtrip(tmp_path):
    p = tmp_path / "_basket.json"
    bk.save(p, D1, {"A", "B"}, config_version=2)
    b = bk.load(p)
    assert b.rebalanced_at == D1 and b.members == {"A", "B"}
    assert b.config_version == 2


def test_legacy_member_list_is_readable(tmp_path):
    """구버전 _last_members.json 은 티커 배열뿐이었다. 읽되 리밸런스 시점은 모른다."""
    p = tmp_path / "_last_members.json"
    p.write_text(json.dumps(["A", "B"]), encoding="utf-8")
    b = bk.load(p)
    assert b.members == {"A", "B"}
    assert not b.exists          # 시점을 모르면 '확정된 적 없음'으로 취급


def test_drift_reports_both_directions():
    b = bk.Basket(rebalanced_at=D1, members={"A", "B", "C"})
    enter, exit_ = b.drift({"B", "C", "D"})
    assert enter == {"D"} and exit_ == {"A"}


def test_scan_message_says_basket_unchanged():
    b = bk.Basket(rebalanced_at=D1, members={"A", "B"})
    msg = bk.describe(b, {"A", "C"}, D2, rebalancing=False)
    assert "스캔" in msg and "바스켓은 그대로" in msg
    assert "14일 전" in msg      # 기준 시점이 얼마나 오래됐는지 보여야 한다


def test_scan_without_basket_warns_hysteresis_is_off():
    """바스켓이 없으면 히스테리시스가 안 걸린다 — 조용히 넘어가면 안 된다."""
    msg = bk.describe(bk.Basket(), {"A"}, D1, rebalancing=False)
    assert "히스테리시스가 적용되지 않았다" in msg


def test_rebalance_message_shows_turnover():
    b = bk.Basket(rebalanced_at=D1, members={"A", "B", "C"})
    msg = bk.describe(b, {"B", "C", "D"}, D2, rebalancing=True)
    assert "리밸런스" in msg and "진입 1" in msg and "이탈 1" in msg


# ── CLI 배선 ─────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]


def test_scan_does_not_write_the_basket():
    """스캔이 바스켓을 갱신하면 기준이 매일 어제로 밀려 분리한 의미가 없다."""
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    assert "if a.rebalance:" in src
    i = src.index("if a.rebalance:")
    assert "bk.save(" in src[i:i + 200], "bk.save 가 --rebalance 밖에서 호출된다"
    assert src.count("bk.save(") == 1, "바스켓 저장 지점이 여러 곳이면 스캔이 새어든다"


def test_hysteresis_reference_is_the_basket_not_last_run():
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    assert "previous_members=current_basket.members" in src


def test_manifest_records_drift():
    """표류를 기록하지 않으면 '언제부터 벌어졌나'에 답할 수 없다."""
    mf = REPO / "data/screens/2026-08-06/manifest.json"
    if not mf.exists():
        pytest.skip("스크린 산출물 없음")
    d = json.loads(mf.read_text(encoding="utf-8"))
    assert "basket" in d
    for k in ("mode", "reference", "entry_candidates", "exit_candidates"):
        assert k in d["basket"], f"매니페스트 basket 에 {k} 가 없다"
    assert d["basket"]["mode"] in ("scan", "rebalance")


# ── 골든셋 코호트 가드 ───────────────────────────────────────────────
def test_cohort_overlap_threshold_is_declared():
    import pipeline.cli as cli
    assert 0.5 <= cli.MIN_COHORT_OVERLAP <= 1.0


def test_golden_blocks_on_cohort_drift():
    """라벨은 티커 단위다. 바스켓이 바뀐 뒤 측정하면 오분류율이
    '데이터 품질'이 아니라 '오늘의 종목 구성'을 잰다. 조용히 숫자를 내면
    그 숫자를 신뢰하게 되므로 막는다."""
    src = (REPO / "src/pipeline/cli.py").read_text(encoding="utf-8")
    assert "cover < MIN_COHORT_OVERLAP" in src
    assert "allow_cohort_drift" in src
    i = src.index("cover < MIN_COHORT_OVERLAP")
    assert "return 1" in src[i:i + 700], "겹침 미달인데 종료 코드가 0 이면 CI 가 못 잡는다"
