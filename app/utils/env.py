# app/utils/env.py
from __future__ import annotations
import os
from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        dotenv_path = _find_env_file()
        if dotenv_path:
            load_dotenv(dotenv_path=str(dotenv_path), override=False)
        else:
            load_dotenv(override=False)
        return
    except Exception:
        dotenv_path = _find_env_file()
        if dotenv_path and dotenv_path.exists():
            _basic_load_dotenv(dotenv_path)


def _find_env_file() -> Path | None:
    candidates = []
    here = Path(__file__).resolve()
    app_dir = here.parent.parent
    root = app_dir.parent

    candidates.append(root / ".env")
    candidates.append(app_dir / ".env")
    candidates.append(Path.cwd() / ".env")

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _basic_load_dotenv(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)
