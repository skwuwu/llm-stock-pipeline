"""스크린 레지스트리 — 하나의 엔진(L0~L7)을 여러 설정으로 병행 운용한다.

저평가 하나로는 투자 아이디어가 되지 않는다. 딥밸류가 구조적으로 못 잡는
집합이 있다 — 실측(2026-08-06): GARP 통과 28종목 중 딥밸류와 겹치는 종목은
**0개**였다(한국카본 ROE 16.4%/PER 12.5, HD현대마린엔진 ROE 35.9%/PER 10.7).
싼 것만 보는 스크린은 이들을 영영 보지 못한다.

엔진은 그대로 두고 **L3 게이트 설정만 갈아끼운다.** 그래서 이 모듈이 하는 일은
설정 파일과 산출물 경로를 스크린 이름으로 묶는 것뿐이다.

경로 규약
─────────
스크린마다 다른 것 (섞이면 뒤에 돌린 스크린이 앞엣것을 덮어쓴다):
    screens/{screen}/{as_of}/   survivors, why_excluded, manifest
    screens/{screen}/_basket.json
    enrich/{screen}/{as_of}/    evidence pack
    llm/{screen}/tags_{as_of}.json
    verify/{screen}/{as_of}/    verdicts, golden_metrics
    out/{screen}/{as_of}/       digest

스크린과 무관한 것 (공유해야 이득인 것):
    derived/metrics_{as_of}.parquet   전 종목 대상이라 스크린을 타지 않는다
    raw/**                            원천은 하나다
    llm/tags/                         **pack_hash 키 캐시.** 같은 종목이 두
                                      스크린에 걸리면 두 번째는 API 호출 0회다.
                                      네임스페이스로 가르면 그 이득이 사라진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

# 설정 파일명만 여기 등록하면 CLI 전체에서 --screen 으로 선택된다.
SCREENS: dict[str, str] = {
    "deep_value": "kr_deep_value.yaml",
    "garp": "kr_garp.yaml",
    "quality_fcf": "kr_quality_fcf.yaml",
    # 테마 우선. 밸류에이션 게이트가 없어 다른 셋과 성격이 완전히 다르다 —
    # 바스켓이 아니라 조사 대상 목록이다.
    "theme_hunt": "kr_theme_hunt.yaml",
}
DEFAULT_SCREEN = "deep_value"

# 골든셋 라벨은 **티커 단위**라 스크린마다 다른 파일이어야 한다.
# 실측: GARP 통과 40종목과 딥밸류 라벨 62종목의 겹침은 0 이다. 한 파일을
# 공유하면 오분류율이 '분류 품질'이 아니라 '두 스크린의 종목 구성 차이'를 잰다.
# 값이 없는 스크린은 golden 이 **명확히 거절**한다 — 조용히 0% 를 내지 않는다.
#
# **라벨은 티커당 한 곳에만 둔다.** 같은 티커를 두 파일에 두면 어느 쪽이
# 정답인지 갈리고 수정이 한쪽에만 반영된다. 그래서 스크린은 파일 하나가
# 아니라 **여러 파일을 합쳐** 자기 코호트를 덮는다 — quality_fcf 통과 40종목
# 중 9종목은 딥밸류와 겹쳐 그쪽 파일에 이미 라벨이 있다.
GOLDEN_LABELS: dict[str, list[str]] = {
    "deep_value": ["kr_core_themes_v1.jsonl"],
    "garp": ["kr_growth_themes_v1.jsonl", "kr_core_themes_v1.jsonl"],
    "quality_fcf": ["kr_growth_themes_v1.jsonl", "kr_core_themes_v1.jsonl"],
    # 라벨 없음. 300종목을 라벨링하지 않았으므로 오분류율을 잴 수 없고,
    # golden 은 조용히 0% 를 내는 대신 거절한다.
    "theme_hunt": [],
}


class UnknownScreenError(ValueError):
    pass


@dataclass(frozen=True)
class ScreenPaths:
    """스크린 하나의 설정·산출물 위치. CLI 는 경로를 직접 조립하지 않는다."""

    screen: str
    repo: Path
    data: Path

    # ── 설정 ──────────────────────────────────────────────────────
    @property
    def config(self) -> Path:
        return self.repo / "configs" / "screen" / SCREENS[self.screen]

    # ── 스크린별 산출물 ───────────────────────────────────────────
    def screen_dir(self, as_of: date | str) -> Path:
        return self.data / "screens" / self.screen / str(as_of)

    @property
    def basket(self) -> Path:
        return self.data / "screens" / self.screen / "_basket.json"

    def enrich_dir(self, as_of: date | str) -> Path:
        return self.data / "enrich" / self.screen / str(as_of)

    def tags(self, as_of: date | str) -> Path:
        return self.data / "llm" / self.screen / f"tags_{as_of}.json"

    def verify_dir(self, as_of: date | str) -> Path:
        return self.data / "verify" / self.screen / str(as_of)

    def out_dir(self, as_of: date | str) -> Path:
        return self.data / "out" / self.screen / str(as_of)

    # ── 공유 산출물 ───────────────────────────────────────────────
    def metrics(self, as_of: date | str) -> Path:
        """전 종목 파생지표. 스크린을 타지 않는다 — derive 는 한 번만 돌린다."""
        return self.data / "derived" / f"metrics_{as_of}.parquet"

    @property
    def golden(self) -> list[Path]:
        """이 스크린의 골든셋 라벨 파일들. 빈 목록이면 호출자가 거절해야 한다."""
        return [self.repo / "tests" / "golden" / fn
                for fn in GOLDEN_LABELS.get(self.screen) or []]

    @property
    def tag_cache(self) -> Path:
        """pack_hash 키 LLM 캐시. **스크린 간 공유가 목적이다.**"""
        return self.data / "llm" / "tags"


def resolve(screen: str | None, repo: Path, data: Path) -> ScreenPaths:
    name = screen or DEFAULT_SCREEN
    if name not in SCREENS:
        raise UnknownScreenError(
            f"모르는 스크린 '{name}'. 등록된 것: {', '.join(sorted(SCREENS))}")
    return ScreenPaths(screen=name, repo=repo, data=data)


def add_screen_arg(parser) -> None:
    """스크린을 인자로 받는 모든 서브커맨드에 동일하게 붙인다."""
    parser.add_argument(
        "--screen", default=DEFAULT_SCREEN, choices=sorted(SCREENS),
        help=f"L3 게이트 설정 (기본 {DEFAULT_SCREEN}). 산출물 경로도 함께 갈린다.")
