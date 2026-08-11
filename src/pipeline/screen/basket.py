"""확정 바스켓 — 스캔 주기와 리밸런스 주기를 분리한다.

히스테리시스는 '직전 **리밸런스**' 를 기준으로 해야 한다. '직전 실행' 을 쓰면
일별로 돌릴 때 기준이 매일 어제로 밀려, 어제 편입된 종목이 오늘도 느슨하게
평가되고 그게 매일 이어져 **바스켓이 서서히 표류한다.** 임계값 근처에서
들락거리는 종목을 막으려고 넣은 장치가 정반대로 작동하는 셈이다.

  스캔(일별)      가격 갱신 → 지표 → 게이트 → **후보 변동만 보고**. 바스켓 불변.
  리밸런스(주/월)  그 시점에 바스켓을 확정하고 enrich/tag/verify/golden 을 돌린다.

바스켓을 확정하는 것이 왜 중요한가: 골든셋 라벨은 티커 단위라, 코호트가 매일
바뀌면 오분류율이 '데이터 품질' 이 아니라 '오늘의 종목 구성' 을 재게 된다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class Basket:
    """확정된 스크린 바스켓. 리밸런스에서만 갱신된다."""

    rebalanced_at: date | None = None
    members: set[str] = field(default_factory=set)
    config_version: int | None = None

    @property
    def exists(self) -> bool:
        return self.rebalanced_at is not None

    def drift(self, current: set[str]) -> tuple[set[str], set[str]]:
        """(신규 진입 후보, 이탈 후보). 스캔이 보고하는 값."""
        return current - self.members, self.members - current


def load(path: Path) -> Basket:
    if not path.exists():
        return Basket()
    d = json.loads(path.read_text(encoding="utf-8"))
    # 구버전: 티커 배열만 있던 _last_members.json
    if isinstance(d, list):
        return Basket(rebalanced_at=None, members=set(d))
    at = d.get("rebalanced_at")
    return Basket(
        rebalanced_at=date.fromisoformat(at) if at else None,
        members=set(d.get("members") or []),
        config_version=d.get("config_version"),
    )


def save(path: Path, as_of: date, members: set[str],
         config_version: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "rebalanced_at": str(as_of),
        "members": sorted(members),
        "config_version": config_version,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def describe(basket: Basket, current: set[str], as_of: date,
             rebalancing: bool) -> str:
    """스캔·리밸런스 결과를 한 문단으로. 표류를 눈에 보이게 하는 게 목적이다."""
    if rebalancing:
        if not basket.exists:
            return f"리밸런스: 바스켓 신규 확정 {len(current)}종목 @ {as_of}"
        enter, exit_ = basket.drift(current)
        return (f"리밸런스: {len(basket.members)} → {len(current)}종목 "
                f"(직전 {basket.rebalanced_at}) | 진입 {len(enter)} 이탈 {len(exit_)}")
    if not basket.exists:
        return (f"스캔: 확정 바스켓이 없다 — 히스테리시스가 적용되지 않았다. "
                f"`screen --rebalance` 로 먼저 확정할 것.")
    enter, exit_ = basket.drift(current)
    days = (as_of - basket.rebalanced_at).days
    return (f"스캔: 확정 바스켓 {len(basket.members)}종목 "
            f"({basket.rebalanced_at}, {days}일 전) 대비 "
            f"진입후보 {len(enter)} 이탈후보 {len(exit_)} — **바스켓은 그대로**")
