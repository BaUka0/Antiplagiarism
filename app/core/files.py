from __future__ import annotations

import re
import os
import shutil
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4


def _sanitize_stem(filename: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    stem = stem.strip("._-")
    return stem or "document"


def build_upload_path(base_dir: str | Path, filename: str, kind: str) -> Path:
    base_path = Path(base_dir)
    safe_name = Path(filename or "document.pdf").name
    suffix = Path(safe_name).suffix.lower() or ".pdf"
    stem = _sanitize_stem(safe_name)
    target_dir = base_path / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{kind}_{stem}_{uuid4().hex[:12]}{suffix}"


def write_stream_to_path(source: BinaryIO, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as buffer:
        shutil.copyfileobj(source, buffer)


def remove_file(path: str | Path | None) -> None:
    if not path:
        return
    target = Path(path)
    if not target.exists():
        return

    with suppress(FileNotFoundError, PermissionError, OSError):
        target.unlink()
        return

    with suppress(Exception):
        os.chmod(target, 0o666)
        target.unlink()
