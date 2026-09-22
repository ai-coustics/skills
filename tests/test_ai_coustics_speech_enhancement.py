import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
SCRIPTS_DIR = REPO / "skills/ai-coustics-speech-enhancement/scripts"
sys.path.insert(0, str(SCRIPTS_DIR))  # so `from _aic_models import ...` resolves

_spec = importlib.util.spec_from_file_location("enhance", SCRIPTS_DIR / "enhance.py")
enhance = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = enhance
_spec.loader.exec_module(enhance)

MANIFEST = {"models": {
    "quail-vf-2.2-l-16khz": {"versions": {"v7": {}}},
    "quail-vf-2.2-s-16khz": {"versions": {"v7": {}}},
    "quail-vf-2.1-l-48khz": {"versions": {"v7": {}}},
    "quail-ms-l-16khz": {"versions": {"v7": {}}},
    "quail-l-16khz": {"versions": {"v7": {}}},
    "rook-ms-s-16khz": {"versions": {"v7": {}}},
    "rook-ms-l-48khz": {"versions": {"v7": {}}},
    "vad-ms-2.1-xxs-16khz": {"versions": {"v7": {}}},
    "tyto-1.1-l-16khz": {"versions": {"v7": {}}},
}}


@pytest.fixture
def online(monkeypatch):
    am = sys.modules["_aic_models"]
    monkeypatch.setattr(am, "fetch_manifest", lambda: MANIFEST)
    monkeypatch.setattr(am, "compatible_version", lambda: "v7")


def test_default_is_a_family_spec_not_a_pinned_version():
    assert enhance.DEFAULT_MODEL == "quail-vf-l-16khz"
    assert sys.modules["_aic_models"].parse_model_id(enhance.DEFAULT_MODEL).version is None


def test_alias_to_spec():
    f = enhance.alias_to_spec
    assert f("vf") == "quail-vf-l-16khz"
    assert f("VF S") == "quail-vf-s-16khz"
    assert f("vf 48k") == "quail-vf-l-48khz"
    assert f("ms") == "quail-ms-l-16khz"
    assert f("rook s") == "rook-ms-s-16khz"
    assert f("rook 48k") == "rook-ms-l-48khz"
    # Pre-rename ids map onto current products.
    assert f("quail-l-16khz") == "quail-ms-l-16khz"
    assert f("rook-s-48khz") == "rook-ms-s-48khz"
    # Anything else passes through for the resolver.
    assert f("quail-vf-2.2-l-16khz") == "quail-vf-2.2-l-16khz"
    assert f("my-custom-model") == "my-custom-model"


def test_resolve_model_name_picks_newest_and_honours_aliases(online):
    assert enhance.resolve_model_name("vf") == "quail-vf-2.2-l-16khz"
    assert enhance.resolve_model_name(enhance.DEFAULT_MODEL) == "quail-vf-2.2-l-16khz"
    assert enhance.resolve_model_name("vf s") == "quail-vf-2.2-s-16khz"
    assert enhance.resolve_model_name("rook s") == "rook-ms-s-16khz"
    assert enhance.resolve_model_name("quail-l-16khz") == "quail-ms-l-16khz"


def test_resolve_model_name_rejects_wrong_kind_and_retired(online):
    with pytest.raises(SystemExit) as e:
        enhance.resolve_model_name("vad-ms-16khz")
    assert "voice-activity-detection" in str(e.value)
    with pytest.raises(SystemExit) as e:
        enhance.resolve_model_name("quail-vf-2.1-l-16khz")
    assert "quail-vf-2.2-l-16khz" in str(e.value)


def test_make_processor_explains_type_errors(monkeypatch):
    class Boom(Exception):
        pass
    monkeypatch.setattr(enhance.aic, "ModelTypeUnsupportedError", Boom)

    def raising(model, key):
        raise Boom()
    monkeypatch.setattr(enhance.aic, "ProcessorAsync", raising)

    class M:
        def get_id(self): return "vad-x"
    with pytest.raises(SystemExit) as e:
        enhance.make_processor(M(), "k")
    assert "not an enhancement model" in str(e.value)


def test_resolve_model_name_offline_never_swaps_family(monkeypatch, capsys):
    am = sys.modules["_aic_models"]
    monkeypatch.setattr(am, "fetch_manifest", lambda: None)
    # The default family may fall back to the last-known-good id of that family ...
    assert enhance.resolve_model_name("vf") == enhance.FALLBACK_MODEL
    assert "without checking the catalog" in capsys.readouterr().err
    # ... but a different family must not silently become Quail Voice Focus.
    with pytest.raises(SystemExit) as e:
        enhance.resolve_model_name("rook")
    assert "offline fallback" in str(e.value) and "--model-path" in str(e.value)


def test_process_file_output_is_aligned_with_input(tmp_path, monkeypatch):
    """The processor delays by D samples; the written file must line up with the input."""
    delay, block = 3, 4
    audio = np.arange(1, 11, dtype=np.float32)  # 10 samples, not a multiple of the block

    class Ctx:
        def reset(self): pass
        def set_parameter(self, *_): pass
        def get_audio_delay(self): return delay

    class DelayLine:
        """Pure delay of `delay` samples, block by block, like the SDK processor."""
        def __init__(self): self.tail = np.zeros(delay, dtype=np.float32)
        async def initialize_async(self, _cfg): pass
        def get_context(self): return Ctx()
        async def process_async(self, buf):
            assert len(buf) == block  # always fed full blocks
            stream = np.concatenate([self.tail, buf])
            self.tail = stream[-delay:].copy()
            return stream[:block]

    class Cfg:
        block_size = block

    written = {}
    monkeypatch.setattr(enhance.aic.ProcessorConfig, "optimal", lambda model, sample_rate: Cfg())
    monkeypatch.setattr(enhance, "load_mono_audio", lambda _p: (audio, 16000))
    monkeypatch.setattr(enhance.sf, "write", lambda path, data, fs: written.update(data=np.asarray(data), fs=fs))

    asyncio.run(enhance._process_file("in.wav", str(tmp_path / "out.wav"), object(), DelayLine(), 1.0))
    np.testing.assert_array_equal(written["data"], audio)
    assert written["fs"] == 16000
