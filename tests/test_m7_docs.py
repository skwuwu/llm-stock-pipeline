"""문서가 코드와 어긋나면 실패한다.

문서는 썩는다. 특히 이 저장소는 실측이 설계를 여러 번 뒤집었고
(등급 폐지, 촉매 도입, 스크린 4종화) 그때마다 DESIGN.md 가 뒤처졌다.
**사람이 지키기로 약속하는 대신 테스트가 지킨다.**

여기서 검사하는 것은 '문장이 예쁜가'가 아니라 **숫자와 이름이 실제와 같은가**다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
ARCH = REPO / "docs/ARCHITECTURE.md"
OPS = REPO / "docs/OPERATIONS.md"
DESIGN = REPO / "docs/DESIGN.md"


def _t(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ── 존재와 링크 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("p", [README, ARCH, OPS, DESIGN])
def test_doc_exists_and_is_substantial(p):
    assert p.exists(), f"{p.name} 이 없다"
    assert len(_t(p)) > 2000, f"{p.name} 이 너무 짧다"


def test_internal_links_resolve():
    """깨진 링크는 '문서가 있다'는 착각만 준다."""
    for doc in (README, ARCH, OPS):
        for target in re.findall(r"\]\((?!https?://)([^)#]+)", _t(doc)):
            assert (doc.parent / target).resolve().exists(), \
                f"{doc.name} → {target} 가 없다"


# ── CLI 와 동기 ──────────────────────────────────────────────────────
def _cli_commands() -> set[str]:
    src = _t(REPO / "src/pipeline/cli.py")
    return set(re.findall(r'sub\.add_parser\("([\w-]+)"', src))


def test_every_cli_command_is_documented():
    """명령을 추가하고 문서를 안 고치면 아무도 그 명령을 모른다."""
    doc = _t(README)
    missing = [c for c in _cli_commands() if f"`{c}`" not in doc]
    assert not missing, f"README 에 없는 명령: {missing}"


def test_readme_does_not_invent_commands():
    """없는 명령을 적어두면 따라 하다 실패한다."""
    cmds = _cli_commands()
    for m in re.findall(r"python -m pipeline\.cli ([\w-]+)", _t(README) + _t(OPS)):
        assert m in cmds, f"문서가 존재하지 않는 명령을 안내한다: {m}"


# ── 설정과 동기 ──────────────────────────────────────────────────────
def test_screen_count_matches_registry():
    from pipeline.screen.registry import SCREENS
    for name in SCREENS:
        assert f"`{name}`" in _t(README), f"README 에 스크린 {name} 이 없다"


def test_theme_count_is_current():
    n = len(yaml.safe_load(
        _t(REPO / "configs/themes/taxonomy_v1.yaml"))["themes"])
    assert f"{n}테마" in _t(README) or f"{n}개" in _t(README), \
        f"문서의 테마 수가 실제({n})와 다르다"


def test_catalyst_counts_are_current():
    cfg = yaml.safe_load(_t(REPO / "configs/catalysts/catalyst_v1.yaml"))
    total = len(cfg["catalysts"])
    enabled = [c["id"] for c in cfg["catalysts"] if c["enabled"]]
    ops = _t(OPS)
    assert f"{total}종" in ops, f"OPERATIONS 의 촉매 총수가 실제({total})와 다르다"
    assert f"활성 {len(enabled)}" in ops
    for cid in enabled:
        assert cid in _t(README), f"README 에 활성 촉매 {cid} 가 없다"


def test_reproducibility_numbers_match_the_constant():
    """이 숫자가 문서에만 있고 코드에 없으면 다음 측정 때 갱신을 잊는다."""
    from pipeline.verify.golden import REPRODUCIBILITY as RP
    for doc in (README, ARCH):
        t = _t(doc)
        assert f"{RP['core_agreement']:.0%}" in t, f"{doc.name}: core 일치율"
        assert f"{RP['verdict_agreement']:.0%}" in t, f"{doc.name}: 판정 일치율"
        assert f"±{RP['recall_noise_pp']}%p" in t, f"{doc.name}: 잡음 폭"


# ── 폐지된 개념이 되살아나지 않는가 ──────────────────────────────────
def test_readme_and_architecture_do_not_promise_tiers():
    """등급(A/B/C)은 폐지됐다. 문서가 그걸 약속하면 코드와 어긋난다.

    DESIGN.md 는 예외다 — 초기 설계 기록이라 보존하되 낡음 배너를 단다.
    """
    for doc in (README, ARCH, OPS):
        t = _t(doc)
        for banned in ("tier A", "tier B", "tier C", "티어 A"):
            # ARCHITECTURE 는 '왜 없앴는지' 를 설명하므로 폐지 맥락은 허용한다
            if banned.lower() in t.lower() and doc is not ARCH:
                pytest.fail(f"{doc.name} 이 폐지된 등급을 언급한다: {banned}")


def test_design_doc_carries_a_staleness_banner():
    """낡은 문서를 지우는 대신 남기려면, 낡았다고 말해야 한다."""
    t = _t(DESIGN)
    head = t[:3000]
    assert "낡" in head, "DESIGN.md 에 낡음 표시가 없다"
    assert "ARCHITECTURE.md" in head, "현재 문서로 가는 안내가 없다"


# ── 한계를 감추지 않는가 ─────────────────────────────────────────────
def test_readme_states_known_limitations():
    """이 저장소의 규율은 '조용히 감추지 않는다' 다. 문서도 같아야 한다."""
    t = _t(README)
    for must in ("한계", "생존편향", "사람 검수 전", "재현성"):
        assert must in t, f"README 가 '{must}' 를 다루지 않는다"


def test_readme_has_a_disclaimer():
    t = _t(README)
    assert "면책" in t
    assert "추천" in t and "아니" in t, "'추천이 아니다' 가 명시돼야 한다"


def test_llm_boundary_is_documented():
    """LLM 에 주가를 넘기지 않는다는 것이 이 설계의 핵심 중 하나다."""
    from pipeline.enrich.evidence import METRIC_KEYS
    t = _t(README)
    assert "METRIC_KEYS" in t
    for k in METRIC_KEYS:
        assert k in t, f"README 의 METRIC_KEYS 목록에 {k} 가 없다"
    assert "per" not in METRIC_KEYS and "close" not in METRIC_KEYS


# ── 리포트 안의 수치가 코드와 어긋나지 않는가 ────────────────────────
def test_digest_does_not_hardcode_reproducibility():
    """실측: '재실행 시 100% 일치'를 각주에 박아뒀다가, 표본을 늘려 96% 로
    정정한 뒤에도 그 문구만 남아 **리포트가 거짓말을 하고 있었다.**
    상수를 읽게 하고 하드코딩을 금지한다."""
    src = (REPO / "src/pipeline/report/digest.py").read_text(encoding="utf-8")
    assert "REPRODUCIBILITY" in src, "재현성 수치를 상수에서 읽지 않는다"
    # 주석은 '한때 100% 라고 박아뒀다'는 이력을 남기므로 검사 대상이 아니다.
    # 실제로 출력되는 코드 줄만 본다.
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    for ln in code:
        if "일치율" in ln or "일치한다" in ln:
            assert re.search(r"_RP\[|REPRODUCIBILITY\[", ln), \
                f"재현성 수치가 하드코딩됐다: {ln.strip()[:70]}"


def test_digest_carries_reliability_metrics():
    """다이제스트만 읽는 사람도 이 문서가 얼마나 맞는지 알아야 한다.
    별도 파일에만 두면 종목만 보고 신뢰도는 안 본다."""
    src = (REPO / "src/pipeline/report/digest.py").read_text(encoding="utf-8")
    for must in ("오분류율", "생존편향", "인용 검증 실패율"):
        assert must in src, f"다이제스트에 {must} 가 없다"


@pytest.mark.parametrize("screen", ["deep_value", "garp", "quality_fcf"])
def test_emitted_digest_shows_misclassification(screen):
    p = REPO / f"data/out/{screen}/2026-08-06/digest.md"
    if not p.exists():
        pytest.skip("다이제스트 없음")
    t = _t(p)
    assert "오분류율" in t, f"{screen} 다이제스트에 오분류율이 없다"
    assert "생존편향" in t, f"{screen} 다이제스트에 생존편향이 없다"
    from pipeline.verify.golden import REPRODUCIBILITY as RP
    assert f"{RP['core_agreement']:.0%}" in t, "재현성 수치가 상수와 다르다"
