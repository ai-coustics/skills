# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aic-sdk>=3.1.0",
#   "certifi",
#   "numpy",
#   "soundfile",
# ]
# ///
"""
Detect voice activity in audio files with the ai-coustics VAD models via the AIC SDK.

Two VAD product lines exist:

  VAD Multi Speaker (alias 'ms', family vad-ms-*)  fires on any audible speech; robust
                                                   to noise, music and far-field speakers.
  VAD Voice Focus   (alias 'vf', family vad-vf-*)  fires on the primary speaker only and
                                                   ignores interfering/background speech.

The concrete artifact is resolved at run time against the public manifest and the
installed SDK's compatible artifact version, so 'ms' always means the newest VAD Multi
Speaker the SDK can load. No SDK or model generation is pinned. Block length,
prediction delay and parameter defaults are all read from the model.

The model outputs a speech probability per block. The SDK thresholds it with
`sensitivity` and smooths it with `speech_hold_duration` / `minimum_speech_duration`
to produce the speech-detected decision. This script records both, per block, and
turns the decisions into speech segments. The audio is never modified.

Usage (single file):
  uv run detect.py call.wav
  uv run detect.py call.wav --sensitivity 0.6 --out-json vad.json --out-csv segments.csv

Usage (folder):
  uv run detect.py /path/to/audio_dir/ --out-csv segments.csv

Options:
  --model-name NAME     'ms' | 'vf' | family spec | exact VAD artifact id (default: ms)
  --model-path PATH     Local .aicmodel file (takes precedence over --model-name)
  --list-models         Print the VAD models the installed SDK can load, then exit
  --sensitivity F       Probability threshold 0.0-1.0; higher = fewer detections
                        (default: read from the model file)
  --speech-hold F       Seconds speech stays "on" after it stops (default: model)
  --min-speech F        Seconds speech must persist before it counts (default: model)
  --print-blocks        Also print one line per block (probability + decision)
  --license-key KEY     Override AIC_SDK_LICENSE_KEY / AIC_LICENSE_KEY / AIC_SDK_LICENSE
  --out-json PATH       Write per-block probabilities, decisions and segments as JSON
  --out-csv PATH        Write speech segments as CSV (file, segment, start_sec, end_sec, duration_sec)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import aic_sdk as aic

from _aic_models import (
    LICENSE_ENV_VARS,
    SUPPORTED_EXTS,
    ModelResolutionError,
    compatible_version,
    download_model,
    fetch_manifest,
    format_catalog,
    load_mono_audio,
    resolve_license,
    resolve_model,
    resolve_or_fallback,
)

MODEL_KIND = "vad"
DEFAULT_MODEL = "vad-ms-16khz"            # family spec: newest VAD Multi Speaker at 16 kHz
FALLBACK_MODEL = "vad-ms-2.1-xxs-16khz"   # used only when the manifest cannot be read

# Friendly aliases -> family spec (version left out, resolved to the newest at run time).
ALIASES: dict[str, str] = {
    "ms": "vad-ms-16khz",
    "multi speaker": "vad-ms-16khz",
    "multi-speaker": "vad-ms-16khz",
    "vad ms": "vad-ms-16khz",
    "vad": "vad-ms-16khz",
    "vf": "vad-vf-16khz",
    "voice focus": "vad-vf-16khz",
    "voice-focus": "vad-vf-16khz",
    "vad vf": "vad-vf-16khz",
}

# Pre-rename ids that still resolve in the manifest but alias a current product
# (see docs: Model Naming Changes). Hidden from --list-models to avoid confusion.
LEGACY_IDS = frozenset({"vad-2.1-xxs-16khz"})

PARAM_FLAGS = (
    # (cli attribute, SDK parameter name, display name)
    ("sensitivity", "Sensitivity", "sensitivity"),
    ("speech_hold", "SpeechHoldDuration", "speech_hold_duration"),
    ("min_speech", "MinimumSpeechDuration", "minimum_speech_duration"),
)


def alias_to_spec(name: str) -> str:
    """Map a friendly alias to a family spec; anything else passes through unchanged."""
    return ALIASES.get(name.strip().lower(), name.strip())


def resolve_model_name(name: str) -> str:
    try:
        return resolve_or_fallback(alias_to_spec(name), FALLBACK_MODEL, kind=MODEL_KIND)
    except ModelResolutionError as e:
        sys.exit(f"{e}\nRun with --list-models to see what the installed SDK can load.")


def list_models() -> None:
    manifest = fetch_manifest()
    if manifest is None:
        sys.exit("Could not read the model manifest. Check network access to artifacts.ai-coustics.io.")
    version = compatible_version()
    print(format_catalog(manifest, version, kind=MODEL_KIND, hide=LEGACY_IDS))

    def current(alias: str) -> str:
        try:
            return resolve_model(alias_to_spec(alias), manifest, version, kind=MODEL_KIND)
        except ModelResolutionError as e:
            return f"nothing ({e})"

    print(f"\nSDK {aic.get_sdk_version()} loads artifact {version}. "
          f"'ms' currently resolves to '{current('ms')}', 'vf' to '{current('vf')}'.")


def load_model(model_name: str, model_path: str | None) -> tuple[aic.Model, str]:
    """Return (model, label) where label is the artifact id or local file stem."""
    if model_path:
        if not os.path.exists(model_path):
            sys.exit(f"Model file not found: {model_path}")
        return aic.Model.from_file(model_path), Path(model_path).stem
    artifact_id = resolve_model_name(model_name)
    print(f"model: {artifact_id} (resolved from '{model_name}'; SDK {aic.get_sdk_version()}, "
          f"artifact {compatible_version()})", file=sys.stderr)
    return aic.Model.from_file(download_model(artifact_id)), artifact_id


# -- Pure helpers (unit-tested, no SDK) --------------------------------------

def segments_from_decisions(
    decisions: list[bool], block_seconds: float, delay_seconds: float, duration: float,
) -> list[tuple[float, float]]:
    """Contiguous runs of speech=True -> (start, end) in seconds on the input timeline.

    The VAD's published decision lags its input by the prediction delay, so block i
    describes audio ending at (i+1)*block - delay. Timestamps are clipped to [0, duration].
    """
    segments: list[tuple[float, float]] = []
    start_idx: int | None = None
    for i, on in enumerate([*decisions, False]):
        if on and start_idx is None:
            start_idx = i
        elif not on and start_idx is not None:
            start = max(0.0, start_idx * block_seconds - delay_seconds)
            end = min(duration, i * block_seconds - delay_seconds)
            if end > start:
                segments.append((start, end))
            start_idx = None
    return segments


@dataclass
class FileResult:
    name: str
    duration: float
    sample_rate: int
    block_seconds: float
    delay_seconds: float
    probabilities: list[float]
    decisions: list[bool]
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def speech_seconds(self) -> float:
        return sum(e - s for s, e in self.segments)

    @property
    def speech_fraction(self) -> float:
        return self.speech_seconds / self.duration if self.duration > 0 else 0.0

    @property
    def first_onset(self) -> float | None:
        return self.segments[0][0] if self.segments else None

    @property
    def longest_segment(self) -> float:
        return max((e - s for s, e in self.segments), default=0.0)


def summarize(name: str, probabilities: list[float], decisions: list[bool],
              duration: float, sample_rate: int, block_seconds: float,
              delay_seconds: float) -> FileResult:
    res = FileResult(name, duration, sample_rate, block_seconds, delay_seconds,
                     probabilities, decisions)
    res.segments = segments_from_decisions(decisions, block_seconds, delay_seconds, duration)
    return res


# -- SDK-facing analysis -------------------------------------------------------

def apply_parameters(ctx, args) -> dict[str, float]:
    """Set any user-supplied VAD parameters, then return the values in effect."""
    for attr, sdk_name, _ in PARAM_FLAGS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        try:
            ctx.set_parameter(getattr(aic.VadParameter, sdk_name), float(value))
        except aic.ParameterOutOfRangeError as e:
            sys.exit(f"--{attr.replace('_', '-')} {value} is out of range for this model: {e}")
    return {
        display: round(float(ctx.get_parameter(getattr(aic.VadParameter, sdk_name))), 4)
        for _, sdk_name, display in PARAM_FLAGS
    }


def analyze_file(model: aic.Model, license_key: str, path: Path, args,
                 model_label: str = "") -> tuple[FileResult, dict[str, float]]:
    samples, sample_rate = load_mono_audio(path)
    config = aic.ProcessorConfig.optimal(model, sample_rate=sample_rate)
    try:
        vad = aic.Vad(model, license_key, config)
    except aic.ModelTypeUnsupportedError:
        sys.exit(
            f"'{model_label or model.get_id()}' is not a VAD model, so the SDK's Vad refuses it. "
            "Enhancement models belong to ai-coustics-speech-enhancement and Tyto to "
            "ai-coustics-audio-insight. Run with --list-models to see VAD models."
        )
    ctx = vad.get_context()
    ctx.reset()
    params = apply_parameters(ctx, args)

    block_size = config.block_size
    block_seconds = block_size / sample_rate
    delay_seconds = ctx.get_prediction_delay() / sample_rate

    probabilities: list[float] = []
    decisions: list[bool] = []
    buf = np.zeros(block_size, dtype=np.float32)
    for start in range(0, len(samples), block_size):
        chunk = samples[start:start + block_size]
        buf[:] = 0.0
        buf[:len(chunk)] = chunk
        vad.process(buf)
        probabilities.append(float(ctx.raw_vad_probability()))
        decisions.append(bool(ctx.is_speech_detected()))

    # The published decision lags its input by the prediction delay, so the last
    # `delay` samples of audio have no decision yet. Flush them with silent blocks;
    # segments_from_decisions() clips the resulting timestamps to the file duration.
    buf[:] = 0.0
    for _ in range(math.ceil(ctx.get_prediction_delay() / block_size)):
        vad.process(buf)
        probabilities.append(float(ctx.raw_vad_probability()))
        decisions.append(bool(ctx.is_speech_detected()))

    duration = len(samples) / sample_rate
    return summarize(path.name, probabilities, decisions, duration, sample_rate,
                     block_seconds, delay_seconds), params


# -- Output --------------------------------------------------------------------

def fmt_params(params: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in params.items())


def print_header(model_id: str, params: dict[str, float], res: FileResult) -> None:
    print(f"# ai-coustics VAD ({model_id}); parameters in effect: {fmt_params(params)}")
    print(f"# {res.block_seconds * 1000:.0f} ms blocks, prediction delay {res.delay_seconds * 1000:.0f} ms "
          "(timestamps already compensated); audio is not modified.")


def print_single_file(res: FileResult, model_id: str, params: dict[str, float], print_blocks: bool) -> None:
    print_header(model_id, params, res)
    print(f"\n{res.name}: {res.duration:.2f}s @ {res.sample_rate} Hz, {len(res.decisions)} blocks")
    if print_blocks:
        for i, (p, d) in enumerate(zip(res.probabilities, res.decisions)):
            t = max(0.0, i * res.block_seconds - res.delay_seconds)
            print(f"  blk {i:>5} [{t:>7.3f}s]  p={p:.3f}  {'SPEECH' if d else '-'}")
    print(f"\n  {'#':>3}  {'start':>8}  {'end':>8}  {'dur':>7}")
    for n, (s, e) in enumerate(res.segments, 1):
        print(f"  {n:>3}  {s:>8.2f}  {e:>8.2f}  {e - s:>7.2f}")
    if not res.segments:
        print("  (no speech detected)")
    onset = f"{res.first_onset:.2f}s" if res.first_onset is not None else "n/a"
    print(
        f"\n  speech  {res.speech_seconds:.2f}s of {res.duration:.2f}s ({res.speech_fraction:.0%})  "
        f"segments={len(res.segments)}  first_onset={onset}  longest={res.longest_segment:.2f}s  "
        f"mean_p={np.mean(res.probabilities):.3f}"
    )


def print_folder(results: list[FileResult], model_id: str, params: dict[str, float]) -> None:
    print_header(model_id, params, results[0])
    name_w = max(len(r.name) for r in results)
    print()
    for i, r in enumerate(results, 1):
        onset = f"{r.first_onset:>6.2f}s" if r.first_onset is not None else "   n/a "
        print(
            f"[{i:>3}/{len(results)}] {r.name:<{name_w}}  "
            f"speech={r.speech_fraction:>4.0%} ({r.speech_seconds:>6.1f}s/{r.duration:>6.1f}s)  "
            f"segments={len(r.segments):>3}  onset={onset}  mean_p={np.mean(r.probabilities):.3f}"
        )
    total = sum(r.duration for r in results)
    speech = sum(r.speech_seconds for r in results)
    silent = [r.name for r in results if not r.segments]
    print(f"\n=== {len(results)} files: {speech:.1f}s speech in {total:.1f}s audio "
          f"({speech / total:.0%}), {sum(len(r.segments) for r in results)} segments ===")
    if silent:
        print(f"no speech detected in {len(silent)}: {', '.join(silent[:5])}{'…' if len(silent) > 5 else ''}")


def write_json(out: Path, model_id: str, params: dict[str, float], results: list[FileResult]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_id,
        "parameters": params,
        "files": [
            {
                "file": r.name,
                "duration_sec": round(r.duration, 3),
                "sample_rate": r.sample_rate,
                "block_sec": round(r.block_seconds, 5),
                "prediction_delay_sec": round(r.delay_seconds, 5),
                "speech_sec": round(r.speech_seconds, 3),
                "speech_fraction": round(r.speech_fraction, 4),
                "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in r.segments],
                "blocks": {
                    "probability": [round(p, 3) for p in r.probabilities],
                    "speech": r.decisions,
                },
            }
            for r in results
        ],
    }
    out.write_text(json.dumps(payload, indent=2))


def write_segments_csv(out: Path, results: list[FileResult]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "segment", "start_sec", "end_sec", "duration_sec"])
        for r in results:
            for n, (s, e) in enumerate(r.segments, 1):
                w.writerow([r.name, n, f"{s:.3f}", f"{e:.3f}", f"{e - s:.3f}"])


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("audio_path", nargs="?", help="Audio file or folder of audio files")
    ap.add_argument("--model-name", default="ms",
                    help="'ms', 'vf', a family spec or an exact VAD artifact id (default: ms)")
    ap.add_argument("--model-path", default=None, help="Local .aicmodel (overrides --model-name)")
    ap.add_argument("--list-models", action="store_true", help="List loadable VAD models and exit")
    ap.add_argument("--sensitivity", type=float, default=None, help="Probability threshold 0-1 (default: model)")
    ap.add_argument("--speech-hold", type=float, default=None, help="Speech hold duration in s (default: model)")
    ap.add_argument("--min-speech", type=float, default=None, help="Minimum speech duration in s (default: model)")
    ap.add_argument("--print-blocks", action="store_true", help="Print one line per block (single-file mode)")
    ap.add_argument("--license-key", default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.audio_path:
        ap.error("audio_path is required (or use --list-models)")

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        sys.exit(f"audio path does not exist: {audio_path}")

    license_key = resolve_license(args.license_key)
    model, model_id = load_model(args.model_name, args.model_path)

    if audio_path.is_file():
        if audio_path.suffix.lower() not in SUPPORTED_EXTS:
            sys.exit(f"Unsupported format: {audio_path.suffix} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")
        files = [audio_path]
    else:
        files = sorted(p for p in audio_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
        if not files:
            sys.exit(f"no audio files in {audio_path} (looked for {sorted(SUPPORTED_EXTS)})")

    results: list[FileResult] = []
    params: dict[str, float] = {}
    failed: list[str] = []
    for f in files:
        try:
            res, params = analyze_file(model, license_key, f, args, model_id)
        except Exception as e:  # unreadable file, SDK error: report, keep going in folder mode
            if audio_path.is_file():
                sys.exit(f"{f.name}: {e}")
            failed.append(f.name)
            print(f"ERROR {f.name}: {e}")
            continue
        results.append(res)
    if not results:
        sys.exit(f"none of the {len(files)} files could be analysed (see ERROR lines)")

    if audio_path.is_file():
        print_single_file(results[0], model_id, params, args.print_blocks)
    else:
        print_folder(results, model_id, params)

    if args.out_json is not None:
        write_json(args.out_json, model_id, params, results)
        print(f"\nwrote JSON to {args.out_json}")
    if args.out_csv is not None:
        write_segments_csv(args.out_csv, results)
        print(f"wrote {sum(len(r.segments) for r in results)} segments to {args.out_csv}")
    if failed:
        sys.exit(f"{len(failed)} of {len(files)} files could not be analysed (see ERROR lines); "
                 "outputs cover the rest.")


if __name__ == "__main__":
    main()
