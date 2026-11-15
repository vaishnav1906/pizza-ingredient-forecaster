"""The in-browser app (docs/app) runs copies of the app's files. Fail if they drift from the source.

Fix a failure with:  python scripts/build_web.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_web import DEST, FILES  # noqa: E402


@pytest.mark.parametrize("src, name", FILES.items())
def test_web_copy_matches_source(src, name):
    assert (DEST / name).read_bytes() == (ROOT / src).read_bytes(), \
        f"docs/app/src/{name} is out of date; run python scripts/build_web.py"
