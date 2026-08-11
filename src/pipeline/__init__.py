"""파이프라인 패키지.

리포 루트 .env 를 임포트 시점에 환경변수로 올린다. anthropic SDK 를 비롯한
여러 클라이언트가 환경변수만 읽으므로, 진입점마다 따로 로드하면 반드시 하나를
빠뜨린다(실측: CLI 는 되는데 `python -m pipeline.themes.validate` 는 키를 못 봤다).
"""
from pathlib import Path as _Path


def load_dotenv(path: _Path | None = None) -> None:
    """이미 설정된 환경변수는 덮어쓰지 않는다."""
    import os
    env = path or _Path(__file__).resolve().parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()
