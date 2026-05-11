from io import BytesIO
import shutil
from pathlib import Path
from uuid import uuid4

from app.core.files import build_upload_path, remove_file, write_stream_to_path


TEST_TMP_ROOT = Path.home() / ".codex" / "memories" / "antiplagiarism" / "test_tmp"


def test_build_upload_path_uses_kind_directory_and_unique_names():
    base_dir = TEST_TMP_ROOT / f"codex_files_{uuid4().hex}"
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        first = build_upload_path(base_dir, "../My File.PDF", kind="checks")
        second = build_upload_path(base_dir, "../My File.PDF", kind="checks")

        assert first != second
        assert first.parent == base_dir / "checks"
        assert first.suffix == ".pdf"
        assert first.name.startswith("checks_My_File_")
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
        shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)


def test_write_and_remove_stream_roundtrip():
    base_dir = TEST_TMP_ROOT / f"codex_files_{uuid4().hex}"
    destination = base_dir / "documents" / "sample.pdf"

    try:
        write_stream_to_path(BytesIO(b"pdf-bytes"), destination)

        assert destination.exists()
        assert destination.read_bytes() == b"pdf-bytes"

        remove_file(destination)
        assert not destination.exists()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
        shutil.rmtree(TEST_TMP_ROOT, ignore_errors=True)
