"""CLI 스모크 테스트.

CLI 가 문법 오류로 깨진 채 백그라운드 배치에 들어가면, 앞 단계는 성공하고
뒤 단계만 조용히 죽는다(실제로 그렇게 시세 수집 단계를 통째로 날렸다).
임포트 가능성과 서브커맨드 파싱을 CI 에서 잡는다.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "pipeline"

EXPECTED_COMMANDS = {
    "init", "ingest-master", "status", "renormalize", "ingest-dart",
    "ingest-prices", "derive", "screen", "enrich", "tag", "verify", "report",
    "golden", "checks", "ingest-dividend", "ingest-holder", "probe-price",
}


def test_every_module_parses():
    """어떤 모듈이든 문법 오류면 즉시 실패."""
    broken = []
    for p in sorted(SRC.rglob("*.py")):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            broken.append(f"{p.relative_to(REPO)}:{e.lineno} {e.msg}")
    assert not broken, "문법 오류:\n" + "\n".join(broken)


def test_cli_imports():
    import pipeline.cli as cli
    assert callable(cli.main)


def test_all_subcommands_registered():
    import pipeline.cli as cli
    parser = _build_parser(cli)
    sub = next(a for a in parser._actions if hasattr(a, "choices") and a.choices)
    assert EXPECTED_COMMANDS <= set(sub.choices), \
        f"누락된 서브커맨드: {EXPECTED_COMMANDS - set(sub.choices)}"


@pytest.mark.parametrize("argv", [
    ["init"], ["status"], ["ingest-master"], ["renormalize"],
    ["ingest-dart", "--year", "2025", "--quarter", "4", "--sample", "10"],
    ["ingest-prices", "--as-of", "2026-08-06", "--with-facts"],
    ["derive", "--as-of", "2026-08-06"],
    ["screen", "--as-of", "2026-08-06", "--target", "40"],
    ["enrich", "--as-of", "2026-08-06", "--limit", "5"],
    ["tag", "--as-of", "2026-08-06", "--dry-run"],
    ["verify", "--as-of", "2026-08-06"],
    ["report", "--as-of", "2026-08-06"],
    ["golden", "--as-of", "2026-08-06"],
    ["checks"],
    ["checks", "--as-of", "2026-08-06"],
    ["ingest-dividend", "--year", "2025", "--limit", "5"],
    ["ingest-holder", "--year", "2025", "--limit", "5"],
    ["probe-price", "--source", "fdr", "--as-of", "2026-08-06"],
    ["ingest-prices", "--as-of", "2026-08-06", "--source", "fdr",
     "--cross-check", "kis"],
    ["screen", "--as-of", "2026-08-06", "--preview"],
    ["screen", "--as-of", "2026-08-06", "--rebalance"],
    ["golden", "--as-of", "2026-08-06", "--allow-cohort-drift"],
    ["screen", "--as-of", "2026-08-06", "--enable", "low_roic", "--disable", "low_roe"],
])
def test_subcommand_args_parse(argv):
    """실행은 하지 않고 인자 파싱만 확인한다(부작용 없음)."""
    import pipeline.cli as cli
    a = _build_parser(cli).parse_args(argv)
    assert callable(a.fn)


def _build_parser(cli):
    """cli.main() 의 파서 구성을 재현한다. main 은 파싱 후 바로 실행하므로 분리 불가."""
    import argparse
    import unittest.mock as mock

    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        captured["parser"] = self
        raise _Stop

    with mock.patch.object(argparse.ArgumentParser, "parse_args", spy):
        try:
            cli.main()
        except _Stop:
            pass
    argparse.ArgumentParser.parse_args = real
    return captured["parser"]


class _Stop(Exception):
    pass


def test_iso_date_parser():
    import pipeline.cli as cli
    assert cli._iso("2026-08-06") == date(2026, 8, 6)
    with pytest.raises(ValueError):
        cli._iso("2026/08/06")
