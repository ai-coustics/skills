import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SCRIPTS_DIR = REPO / "skills/ai-coustics-voice-activity-detection/scripts"

# detect.py imports aic_sdk at module top; it is stubbed in conftest.py.
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location("detect", SCRIPTS_DIR / "detect.py")
detect = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = detect  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(detect)

BLOCK = 0.015
DELAY = 0.030


# ---------------------------------------------------------------------------
# Constants and aliases
# ---------------------------------------------------------------------------

def test_default_model_is_a_family_spec():
    am = sys.modules["_aic_models"]
    parsed = am.parse_model_id(detect.DEFAULT_MODEL)
    assert parsed.family == "vad" and parsed.variant == "ms" and parsed.version is None
    assert detect.FALLBACK_MODEL.startswith("vad-ms-")


def test_alias_to_spec():
    f = detect.alias_to_spec
    assert f("ms") == "vad-ms-16khz"
    assert f("Multi Speaker") == "vad-ms-16khz"
    assert f("vad") == "vad-ms-16khz"
    assert f("vf") == "vad-vf-16khz"
    assert f("voice focus") == "vad-vf-16khz"
    # Exact ids and unknown names pass through for the resolver.
    assert f("vad-ms-2.1-xxs-16khz") == "vad-ms-2.1-xxs-16khz"
    assert f("my-custom-model") == "my-custom-model"


def test_resolve_model_name_live(monkeypatch):
    am = sys.modules["_aic_models"]
    manifest = {"models": {
        "vad-ms-2.1-xxs-16khz": {"versions": {"v7": {}}},
        "vad-ms-2.2-xxs-16khz": {"versions": {"v8": {}}},   # next generation
        "vad-vf-2.0-s-16khz": {"versions": {"v7": {}}},
        "quail-vf-2.2-l-16khz": {"versions": {"v7": {}}},
    }}
    monkeypatch.setattr(am, "fetch_manifest", lambda: manifest)
    monkeypatch.setattr(am, "compatible_version", lambda: "v7")
    assert detect.resolve_model_name("ms") == "vad-ms-2.1-xxs-16khz"
    assert detect.resolve_model_name("vf") == "vad-vf-2.0-s-16khz"
    monkeypatch.setattr(am, "compatible_version", lambda: "v8")
    assert detect.resolve_model_name("ms") == "vad-ms-2.2-xxs-16khz"   # newer SDK, newer model
    with pytest.raises(SystemExit) as e:
        detect.resolve_model_name("quail-vf-l-16khz")
    assert "speech-enhancement" in str(e.value)


def test_supported_extensions_and_license_vars():
    assert set(detect.SUPPORTED_EXTS) == {".wav", ".flac", ".mp3", ".ogg"}  # no AAC in libsndfile
    assert detect.LICENSE_ENV_VARS == ("AIC_SDK_LICENSE_KEY", "AIC_LICENSE_KEY", "AIC_SDK_LICENSE")


# ---------------------------------------------------------------------------
# License resolution
# ---------------------------------------------------------------------------

def test_resolve_license_cli_wins(monkeypatch):
    monkeypatch.setenv("AIC_SDK_LICENSE", "from-env")
    assert detect.resolve_license("from-cli") == "from-cli"


def test_resolve_license_env_fallback(monkeypatch):
    for var in detect.LICENSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AIC_SDK_LICENSE", "env-key")
    assert detect.resolve_license(None) == "env-key"


def test_resolve_license_missing_exits(monkeypatch):
    for var in detect.LICENSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit):
        detect.resolve_license(None)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def test_segments_basic_runs_are_delay_compensated():
    #                 0  1  2  3  4  5  6  7
    decisions = [False, True, True, True, False, False, True, True]
    segs = detect.segments_from_decisions(decisions, BLOCK, DELAY, duration=10.0)
    # blocks 1-3 -> [1*0.015-0.03, 4*0.015-0.03] = [0.0 (clipped), 0.03]
    # blocks 6-7 -> [6*0.015-0.03, 8*0.015-0.03] = [0.06, 0.09]
    assert len(segs) == 2
    assert segs[0] == (0.0, pytest.approx(0.03))
    assert segs[1] == (pytest.approx(0.06), pytest.approx(0.09))


def test_segments_trailing_speech_is_closed_and_clipped_to_duration():
    decisions = [False, False, True, True]
    segs = detect.segments_from_decisions(decisions, 1.0, 0.0, duration=3.5)
    assert segs == [(2.0, 3.5)]


def test_segments_none_when_no_speech():
    assert detect.segments_from_decisions([False] * 5, BLOCK, DELAY, 1.0) == []


def test_segments_drop_empty_after_clipping():
    # A one-block run that the delay pushes entirely before t=0 must not appear.
    segs = detect.segments_from_decisions([True, False], BLOCK, delay_seconds=1.0, duration=5.0)
    assert segs == []


# ---------------------------------------------------------------------------
# Summary / FileResult
# ---------------------------------------------------------------------------

def _result(decisions, probs=None, duration=None, block=1.0, delay=0.0):
    probs = probs if probs is not None else [1.0 if d else 0.0 for d in decisions]
    duration = duration if duration is not None else len(decisions) * block
    return detect.summarize("x.wav", probs, decisions, duration, 16000, block, delay)


def test_summarize_speech_stats():
    res = _result([False, True, True, False, True], block=1.0)
    assert res.segments == [(1.0, 3.0), (4.0, 5.0)]
    assert math.isclose(res.speech_seconds, 3.0)
    assert math.isclose(res.speech_fraction, 0.6)
    assert res.first_onset == 1.0
    assert res.longest_segment == 2.0


def test_summarize_no_speech():
    res = _result([False, False])
    assert res.segments == []
    assert res.speech_fraction == 0.0
    assert res.first_onset is None
    assert res.longest_segment == 0.0


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def test_write_segments_csv(tmp_path):
    out = tmp_path / "nested" / "segments.csv"
    a = _result([True, True, False, True])
    b = _result([False, False])
    b.name = "b.wav"
    detect.write_segments_csv(out, [a, b])
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["file", "segment", "start_sec", "end_sec", "duration_sec"]
    assert rows[1] == ["x.wav", "1", "0.000", "2.000", "2.000"]
    assert rows[2] == ["x.wav", "2", "3.000", "4.000", "1.000"]
    assert len(rows) == 3  # b.wav has no segments


def test_write_json_structure(tmp_path):
    out = tmp_path / "vad.json"
    res = _result([False, True], probs=[0.12345, 0.98765], block=0.5)
    detect.write_json(out, "vad-ms-2.1-xxs-16khz", {"sensitivity": 0.6}, [res])
    data = json.loads(out.read_text())
    assert data["model"] == "vad-ms-2.1-xxs-16khz"
    assert data["parameters"] == {"sensitivity": 0.6}
    f = data["files"][0]
    assert f["file"] == "x.wav"
    assert f["segments"] == [{"start": 0.5, "end": 1.0}]
    assert f["blocks"]["probability"] == [0.123, 0.988]
    assert f["blocks"]["speech"] == [False, True]
    assert f["speech_fraction"] == 0.5


# ---------------------------------------------------------------------------
# analyze_file with a fake SDK
# ---------------------------------------------------------------------------

class _FakeCtx:
    def __init__(self, probs):
        self._probs = list(probs)
        self._i = -1
        self.params = {"Sensitivity": 0.6, "SpeechHoldDuration": 0.1, "MinimumSpeechDuration": 0.0}

    def reset(self): pass
    def get_prediction_delay(self): return 2  # samples
    def step(self): self._i += 1
    def raw_vad_probability(self): return self._probs[self._i]
    def is_speech_detected(self): return self._probs[self._i] > self.params["Sensitivity"]
    def set_parameter(self, p, v): self.params[p] = v
    def get_parameter(self, p): return self.params[p]


def test_analyze_file_end_to_end_with_fake_sdk(monkeypatch):
    import numpy as np

    # 4 audio blocks + 1 flush block for the 2-sample prediction delay (block_size 2).
    probs = [0.1, 0.9, 0.9, 0.2, 0.2]
    ctx = _FakeCtx(probs)

    class FakeVad:
        def __init__(self, model, key, config):
            assert key == "k"
            assert config.block_size == 2
        def get_context(self): return ctx
        def process(self, buf):
            assert buf.dtype == np.float32 and len(buf) == 2
            ctx.step()

    class FakeConfig:
        block_size = 2

    class FakeParam:  # attribute access returns the name itself
        def __getattr__(self, n): return n

    monkeypatch.setattr(detect.aic, "Vad", FakeVad)
    monkeypatch.setattr(detect.aic, "VadParameter", FakeParam())
    monkeypatch.setattr(detect.aic.ProcessorConfig, "optimal", lambda model, sample_rate: FakeConfig())
    # 7 samples at 4 Hz -> 4 blocks of 2 (last one zero-padded); 1.75 s of audio.
    monkeypatch.setattr(detect, "load_mono_audio", lambda p: (np.ones(7, dtype=np.float32), 4))

    args = types.SimpleNamespace(sensitivity=0.5, speech_hold=None, min_speech=None, model_path=None, model_name="ms")
    res, params = detect.analyze_file(object(), "k", Path("a.wav"), args)

    assert params == {"sensitivity": 0.5, "speech_hold_duration": 0.1, "minimum_speech_duration": 0.0}
    assert res.probabilities == probs
    assert res.decisions == [False, True, True, False, False]
    assert math.isclose(res.block_seconds, 0.5)
    assert math.isclose(res.delay_seconds, 0.5)  # 2 samples / 4 Hz
    # blocks 1-2 speech -> [1*0.5-0.5, 3*0.5-0.5] = [0.0, 1.0]
    assert res.segments == [(0.0, pytest.approx(1.0))]
    assert math.isclose(res.duration, 1.75)


def test_analyze_file_flushes_trailing_delay_so_final_speech_is_not_cut(monkeypatch):
    """Speech running to the end of the file must still close at the file duration."""
    import numpy as np

    # 4 audio blocks of 2 samples at 4 Hz (2.0 s), delay 2 samples -> 1 flush block.
    # Speech probability stays high through the flush, so the last real block's
    # decision (published one block late) is still collected.
    probs = [0.1, 0.1, 0.9, 0.9, 0.9]
    ctx = _FakeCtx(probs)

    class FakeVad:
        def __init__(self, model, key, config): pass
        def get_context(self): return ctx
        def process(self, buf): ctx.step()

    class FakeConfig:
        block_size = 2

    class FakeParam:
        def __getattr__(self, n): return n

    monkeypatch.setattr(detect.aic, "Vad", FakeVad)
    monkeypatch.setattr(detect.aic, "VadParameter", FakeParam())
    monkeypatch.setattr(detect.aic.ProcessorConfig, "optimal", lambda model, sample_rate: FakeConfig())
    monkeypatch.setattr(detect, "load_mono_audio", lambda p: (np.ones(8, dtype=np.float32), 4))
    args = types.SimpleNamespace(sensitivity=None, speech_hold=None, min_speech=None, model_path=None, model_name="ms")

    res, _ = detect.analyze_file(object(), "k", Path("a.wav"), args)
    assert len(res.decisions) == 5
    # blocks 2-4 speech -> [2*0.5-0.5, 5*0.5-0.5] = [0.5, 2.0], clipped to duration 2.0
    assert res.segments == [(0.5, 2.0)]
