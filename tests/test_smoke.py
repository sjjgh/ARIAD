"""Isolate the two historical modules with identical import names."""
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant', ['single', 'qk'])
def test_attention_training_and_eval(variant):
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'smoke.py'), '--variant', variant],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
