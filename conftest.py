"""
Stub the heavy runtime packages before any test imports a skill script.
CI installs only pytest, numpy and pyyaml; the AIC SDK needs a license key and a
model download, so tests exercise the pure-Python parts of each script.
"""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "aic_sdk",
    "soundfile",
    "tqdm",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
