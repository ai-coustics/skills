import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SCRIPTS_DIR = REPO / "skills/ai-coustics-audio-insight/scripts"

# score.py imports aic_sdk at module top; it is stubbed in conftest.py.
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location("score", SCRIPTS_DIR / "score.py")
score = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = score
_spec.loader.exec_module(score)


def _result(**dims):
    """Build a fake AnalysisResult-like object with the seven Tyto attrs."""
    full = {d: 0.0 for d in score.DIMENSIONS}
    full.update(dims)
    return types.SimpleNamespace(**full)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_dimensions_and_degradation_dims():
    assert score.DIMENSIONS[0] == "risk_score"
    assert set(score.DIMENSIONS) == {
        "risk_score", "noise", "speaker_reverb", "speaker_loudness",
        "interfering_speech", "codec_degradation", "packet_loss",
    }
    # The neutral level meter and the headline are excluded from the "why" argmax.
    assert "speaker_loudness" not in score.DEGRADATION_DIMS
    assert "risk_score" not in score.DEGRADATION_DIMS
    assert set(score.DEGRADATION_DIMS) == {
        "noise", "speaker_reverb", "interfering_speech", "codec_degradation", "packet_loss",
    }


def test_supported_extensions():
    # libsndfile has no AAC decoder, so .m4a is deliberately absent.
    assert set(score.SUPPORTED_EXTS) == {".wav", ".flac", ".mp3", ".ogg"}


def test_license_env_vars():
    # Same precedence in every skill (AGENTS.md order).
    assert score.LICENSE_ENV_VARS == (
        "AIC_SDK_LICENSE_KEY", "AIC_LICENSE_KEY", "AIC_SDK_LICENSE",
    )


# ---------------------------------------------------------------------------
# License resolution
# ---------------------------------------------------------------------------

def test_resolve_license_cli_key_wins(monkeypatch):
    monkeypatch.setenv("AIC_SDK_LICENSE", "from-env")
    assert score.resolve_license("from-cli") == "from-cli"


def test_resolve_license_env_fallback(monkeypatch):
    for var in score.LICENSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AIC_LICENSE_KEY", "env-key")
    assert score.resolve_license(None) == "env-key"


def test_resolve_license_missing_exits(monkeypatch):
    for var in score.LICENSE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit):
        score.resolve_license(None)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def test_call_aggregates_mean_p95_max_and_frac():
    rows = [
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.1},
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.5},
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.9},
    ]
    agg = score.call_aggregates(rows, threshold=0.30)
    assert math.isclose(agg["mean"]["risk_score"], 0.5)
    assert math.isclose(agg["max"]["risk_score"], 0.9)
    assert agg["p95"]["risk_score"] > 0.8
    # 2 of 3 windows are >= 0.30
    assert math.isclose(agg["frac_degraded"], 2 / 3)


def test_call_aggregates_worst_dim_excludes_loudness():
    # speaker_loudness is highest but neutral → must not be picked as "why".
    rows = [
        {d: 0.0 for d in score.DIMENSIONS}
        | {"speaker_loudness": 0.99, "noise": 0.4, "packet_loss": 0.1},
    ]
    agg = score.call_aggregates(rows, threshold=0.30)
    assert agg["worst_dim"] == "noise"


# ---------------------------------------------------------------------------
# Risk bands
# ---------------------------------------------------------------------------

def test_risk_band_thresholds():
    # Documented bands: Good < 0.30, Warn 0.30-0.50, Bad > 0.50.
    assert score.risk_band(0.20).startswith("🟢")
    assert score.risk_band(0.40).startswith("🟡")
    assert score.risk_band(0.75).startswith("🔴")
    # Boundary: exactly the threshold is not "Good".
    assert not score.risk_band(0.30).startswith("🟢")
    # 0.50 is the top of the warn band.
    assert score.risk_band(0.50).startswith("🟡")
    assert score.risk_band(0.501).startswith("🔴")
    # A custom threshold moves the Good/Warn cut but not the Bad cut.
    assert score.risk_band(0.30, 0.35).startswith("🟢")


# ---------------------------------------------------------------------------
# Dashboard JSON
# ---------------------------------------------------------------------------

def test_dashboard_entry_structure_and_rounding():
    rows = [
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.123456, "noise": 0.5},
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.2, "noise": 0.25},
    ]
    entry = score.dashboard_entry("call.wav", rows, duration=12.3456)
    assert entry["file"] == "call.wav"
    assert entry["duration_sec"] == 12.35  # rounded to 2 dp
    assert set(entry["frames"]) == set(score.DIMENSIONS)
    assert entry["frames"]["risk_score"] == [0.1235, 0.2]  # rounded to 4 dp
    assert len(entry["frames"]["noise"]) == 2


def test_write_dashboard_json_round_trip(tmp_path):
    out = tmp_path / "nested" / "analysis.json"
    rows = [{d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.3}]
    calls = [score.dashboard_entry("a.wav", rows, 5.0)]
    score.write_dashboard_json(out, calls)
    data = json.loads(out.read_text())
    assert data["model"] == "Tyto"
    assert data["calls"][0]["file"] == "a.wav"
    assert data["calls"][0]["frames"]["risk_score"] == [0.3]


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def _agg_with_mean(**means):
    full = {d: 0.0 for d in score.DIMENSIONS}
    full.update(means)
    return {"mean": full}


def test_write_means_csv_round_trip(tmp_path):
    out = tmp_path / "scores.csv"
    per_file = [
        ("a.wav", _agg_with_mean(risk_score=0.1, noise=0.2)),
        ("b.wav", _agg_with_mean(risk_score=0.4, packet_loss=0.6)),
    ]
    score.write_means_csv(out, per_file)
    rows = list(csv.reader(out.open()))
    assert rows[0] == ["", *(f"tyto_{d}" for d in score.DIMENSIONS)]
    assert rows[1][0] == "a.wav"
    risk_idx = score.DIMENSIONS.index("risk_score") + 1
    assert math.isclose(float(rows[1][risk_idx]), 0.1)
    pl_idx = score.DIMENSIONS.index("packet_loss") + 1
    assert math.isclose(float(rows[2][pl_idx]), 0.6)


def test_write_means_csv_creates_parent_dir(tmp_path):
    out = tmp_path / "nested" / "dir" / "scores.csv"
    score.write_means_csv(out, [("x.wav", _agg_with_mean())])
    assert out.exists()


# ---------------------------------------------------------------------------
# analyze_file (analyzer + audio loading mocked)
# ---------------------------------------------------------------------------

def test_analyze_file_builds_rows_and_duration(monkeypatch):
    # 32000 mono samples at 16 kHz → 2.0 s.
    import numpy as np
    monkeypatch.setattr(
        score, "load_mono_audio",
        lambda _p: (np.zeros(32000, dtype=np.float32), 16000),
    )

    class FakeAnalyzer:
        def analyze(self, samples, sample_rate, step_samples):
            assert step_samples == 16000  # 1.0 s hop at 16 kHz
            return [_result(risk_score=0.4, noise=0.7)]

    rows, duration = score.analyze_file(FakeAnalyzer(), Path("x.wav"), step_seconds=1.0)
    assert math.isclose(duration, 2.0)
    assert rows == [{d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.4, "noise": 0.7}]


def test_analyze_file_empty_when_no_windows(monkeypatch):
    import numpy as np
    monkeypatch.setattr(
        score, "load_mono_audio",
        lambda _p: (np.zeros(1000, dtype=np.float32), 16000),
    )

    class FakeAnalyzer:
        def analyze(self, *_a, **_k):
            return []

    rows, _ = score.analyze_file(FakeAnalyzer(), Path("short.wav"), step_seconds=1.0)
    assert rows == []


# ---------------------------------------------------------------------------
# End-to-end printing flows
# ---------------------------------------------------------------------------

def test_print_single_file_reports_band_and_why(capsys):
    rows = [
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.1, "speaker_reverb": 0.8},
        {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.3, "speaker_reverb": 0.6},
    ]
    score.print_single_file("call.wav", rows, duration=6.0, step=1.0, threshold=0.30)
    out = capsys.readouterr().out
    assert "call.wav: 2 window(s)" in out
    assert "win   0 [  0.0s]" in out
    assert "mean=0.200" in out  # mean risk of 0.1 and 0.3
    assert "🟢 Good" in out     # mean risk 0.2 < 0.30
    assert "speaker_reverb" in out  # worst degradation dimension


def test_print_folder_skips_unusable_and_writes_outputs(capsys, tmp_path, monkeypatch):
    files = [Path(f"{n}.wav") for n in ("a", "b", "c")]

    canned = {
        "a.wav": ([{d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.2, "noise": 0.5}], 30.0),
        "b.wav": ([], 3.0),  # too short → skipped
        "c.wav": ([{d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.7, "packet_loss": 0.9}], 15.0),
    }
    monkeypatch.setattr(score, "analyze_file", lambda _an, f, _step: canned[f.name])

    per_file, calls, judge_blocks, failed = score.print_folder(files, step=1.0, threshold=0.30, analyzer=object())
    assert failed == []
    assert len(judge_blocks) == 2 and "Audio quality context" in judge_blocks[0]
    out = capsys.readouterr().out
    assert "skipped: < one 5s window" in out
    assert "Summary across 2 files" in out
    assert [name for name, _ in per_file] == ["a.wav", "c.wav"]
    assert [c["file"] for c in calls] == ["a.wav", "c.wav"]

    # CSV built from the returned per-file aggregates excludes the skipped file.
    out_csv = tmp_path / "scores.csv"
    score.write_means_csv(out_csv, per_file)
    names = [r[0] for r in list(csv.reader(out_csv.open()))[1:]]
    assert names == ["a.wav", "c.wav"]


def test_print_folder_all_skipped_exits(monkeypatch):
    files = [Path("a.wav")]
    monkeypatch.setattr(score, "analyze_file", lambda _an, _f, _step: ([], 2.0))
    with pytest.raises(SystemExit):
        score.print_folder(files, step=1.0, threshold=0.30, analyzer=object())


def test_print_folder_reports_unreadable_file_and_keeps_going(capsys, monkeypatch):
    files = [Path("ok.wav"), Path("broken.wav")]

    def analyze(_an, f, _step):
        if f.name == "broken.wav":
            raise RuntimeError("not an audio file")
        return [{d: 0.0 for d in score.DIMENSIONS} | {"risk_score": 0.2}], 10.0

    monkeypatch.setattr(score, "analyze_file", analyze)
    per_file, calls, _blocks, failed = score.print_folder(files, step=1.0, threshold=0.30, analyzer=object())
    assert [n for n, _ in per_file] == ["ok.wav"] and [c["file"] for c in calls] == ["ok.wav"]
    assert failed == ["broken.wav"]
    out = capsys.readouterr().out
    assert "broken.wav" in out and "ERROR: not an audio file" in out
    assert "Summary across 1 files" in out


# ---------------------------------------------------------------------------
# Version agnosticism: dimensions come from the SDK, the default is a family spec
# ---------------------------------------------------------------------------

def test_default_model_is_family_spec():
    am = sys.modules["_aic_models"]
    assert am.parse_model_id(score.DEFAULT_MODEL).version is None
    assert am.parse_model_id(score.DEFAULT_MODEL).kind == "analysis"


def test_discover_dimensions_from_a_result_type_with_extra_dimension():
    class FutureResult:
        risk_score = 0.0
        noise = 0.0
        speaker_loudness = 0.0
        packet_loss = 0.0
        new_dimension = 0.0     # something a future Tyto adds
    dims = score.discover_dimensions(FutureResult)
    assert dims[0] == "risk_score"
    assert dims[-1] == "new_dimension"          # unknown extras appended
    assert "speaker_reverb" not in dims          # only what the SDK exposes
    # the argmax pool is derived: everything but the headline and neutral meters
    pool = tuple(d for d in dims if d != score.HEADLINE and d not in score.NEUTRAL_DIMS)
    assert pool == ("noise", "packet_loss", "new_dimension")


def test_discover_dimensions_falls_back_when_sdk_is_stubbed():
    # conftest stubs aic_sdk with a MagicMock instance, which is not a type.
    assert score.discover_dimensions(object()) == score.KNOWN_ORDER
    class NoHeadline:
        noise = 0.0
    assert score.discover_dimensions(NoHeadline) == score.KNOWN_ORDER


def test_resolve_model_name_uses_live_manifest(monkeypatch):
    am = sys.modules["_aic_models"]
    manifest = {"models": {"tyto-1.1-l-16khz": {"versions": {"v7": {}}},
                           "tyto-1.2-l-16khz": {"versions": {"v7": {}}}}}
    monkeypatch.setattr(am, "fetch_manifest", lambda: manifest)
    monkeypatch.setattr(am, "compatible_version", lambda: "v7")
    assert score.resolve_model_name(score.DEFAULT_MODEL) == "tyto-1.2-l-16khz"
    with pytest.raises(SystemExit) as e:
        score.resolve_model_name("quail-vf-l-16khz")
    assert "speech-enhancement" in str(e.value)


# ---------------------------------------------------------------------------
# LLM-judge context
# ---------------------------------------------------------------------------

def _row(risk, **dims):
    return {d: 0.0 for d in score.DIMENSIONS} | {"risk_score": risk} | dims


def test_degraded_intervals_merges_overlapping_windows_and_picks_driver():
    # 1 s hop, 5 s windows. Windows 2-4 (noise) and 8 (packet_loss) overlap in time
    # (window 4 spans 4-9 s, window 8 starts at 8 s) -> one interval. Window 14 is a
    # separate stretch (window 8 ends at 13 s).
    rows = [_row(0.1), _row(0.1), _row(0.4, noise=0.7), _row(0.5, noise=0.8), _row(0.35, noise=0.6),
            _row(0.1), _row(0.1), _row(0.1), _row(0.6, packet_loss=0.9), _row(0.1),
            _row(0.1), _row(0.1), _row(0.1), _row(0.1), _row(0.45, speaker_reverb=0.7), _row(0.1)]
    ivs = score.degraded_intervals(rows, step=1.0, threshold=0.30, duration=25.0)
    assert ivs == [
        {"start": 2.0, "end": 13.0, "peak_risk": 0.6, "driver": "noise"},          # 2..8 -> [2, 8+5]
        {"start": 14.0, "end": 19.0, "peak_risk": 0.45, "driver": "speaker_reverb"},
    ]


def test_degraded_intervals_never_overlap_at_one_second_hop():
    # Alternating degraded windows would overlap in time; they must merge.
    rows = [_row(0.5 if i % 2 else 0.1) for i in range(12)]
    ivs = score.degraded_intervals(rows, step=1.0, threshold=0.30, duration=20.0)
    assert len(ivs) == 1 and ivs[0]["start"] == 1.0 and ivs[0]["end"] == 16.0
    # With a coarse hop equal to the window, only consecutive windows merge.
    ivs = score.degraded_intervals(rows, step=5.0, threshold=0.30, duration=100.0)
    assert len(ivs) == 6


def test_degraded_intervals_merge_uses_time_overlap_for_non_dividing_hops():
    # 0.7 s hop: windows 7 hops apart start 4.9 s apart and still overlap by 0.1 s,
    # so they must merge (int(5/0.7) == 7 would have split them into overlapping intervals).
    rows = [_row(0.1)] * 20
    rows[2] = _row(0.5, noise=0.9)
    rows[9] = _row(0.5, noise=0.9)    # 7 hops later
    rows[17] = _row(0.5, noise=0.9)   # 8 hops later: 5.6 s apart, no overlap
    ivs = score.degraded_intervals(rows, step=0.7, threshold=0.30, duration=60.0)
    assert len(ivs) == 2
    assert ivs[0]["start"] == pytest.approx(1.4) and ivs[0]["end"] == pytest.approx(11.3)
    assert ivs[1]["start"] == pytest.approx(11.9)
    assert ivs[0]["end"] <= ivs[1]["start"]  # intervals never overlap


def test_degraded_intervals_clips_to_duration_and_handles_none():
    rows = [_row(0.1), _row(0.9, noise=0.5)]
    ivs = score.degraded_intervals(rows, step=1.0, threshold=0.30, duration=4.5)
    assert ivs == [{"start": 1.0, "end": 4.5, "peak_risk": 0.9, "driver": "noise"}]
    assert score.degraded_intervals([_row(0.1)] * 3, 1.0, 0.30, 7.0) == []


def test_judge_context_markdown_content():
    rows = [_row(0.1)] * 5 + [_row(0.55, packet_loss=0.8)] * 3 + [_row(0.1)] * 4
    md = score.judge_context_markdown("call.wav", rows, duration=16.0, step=1.0,
                                      threshold=0.30, model_label="tyto-x")
    assert md.startswith("### Audio quality context (ai-coustics Tyto, tyto-x)")
    assert "Call: call.wav · 16 s · 12 windows of 5 s" in md
    assert "25% of windows degraded" in md          # 3 of 12
    assert "- 00:05–00:12 (7 s) peak risk 0.55 · driver: packet_loss" in md
    assert "not by the agent" in md
    assert "Do not use it as a tone, empathy or sentiment signal" in md
    assert "Quail Voice Focus isolates the primary speaker" in md


def test_judge_context_markdown_no_degradation():
    md = score.judge_context_markdown("clean.wav", [_row(0.05)] * 6, 10.0, 1.0, 0.30, "tyto-x")
    assert "Degraded intervals: none" in md
    assert "🟢" in md


def test_write_judge_context(tmp_path):
    out = tmp_path / "ctx" / "judge.md"
    score.write_judge_context(out, ["### a\n", "### b\n"])
    assert out.read_text() == "### a\n\n### b\n"
