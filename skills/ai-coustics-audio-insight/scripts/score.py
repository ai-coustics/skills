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
Score audio with the ai-coustics Tyto audio-insight model via the AIC SDK.

Tyto predicts whether audio flowing into a Voice AI stack is likely to cause
downstream failures. It returns, per 5-second window, a headline `risk_score` plus a
set of quality dimensions, all in [0, 1]. For every dimension except the neutral
level meter `speaker_loudness`, higher is worse.

The model is resolved at run time against the public artifact manifest and the
installed SDK's compatible artifact version, so the default is always the newest Tyto
the SDK can load, and the dimensions are read from the SDK's AnalysisResult type
rather than hardcoded. No SDK or model generation is pinned.

Usage (single file):
  uv run score.py call.wav

Usage (folder):
  uv run score.py /path/to/audio_dir/ --out-csv scores.csv --out-json analysis.json

Options:
  --step-seconds F   Hop between the 5-s windows (default: 1.0 -> smooth timeline).
                     Larger values are faster/coarser; the window is always 5 s.
  --threshold F      "Noticeable degradation" threshold for the risk band and the
                     fraction-of-windows feature (default: 0.30, the documented
                     Warn-band cut).
  --model-name NAME  Family spec or exact id (default: tyto-l-16khz = newest Tyto L 16 kHz).
  --model-path PATH  Local .aicmodel file (takes precedence over --model-name).
  --list-models      Print the analysis models the installed SDK can load, then exit.
  --license-key KEY  Override AIC_SDK_LICENSE_KEY / AIC_LICENSE_KEY / AIC_SDK_LICENSE.
  --out-json PATH    Write the call-analysis dashboard JSON (file or folder mode).
  --out-csv PATH     Folder mode only: write per-file means to a scores.csv-style file.
  --out-judge PATH   Write a Markdown "audio quality context" block per call for an LLM
                     judge: degraded intervals with timestamps, driver, and how to
                     interpret them next to the transcript (file or folder mode).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
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

MODEL_KIND = "analysis"
DEFAULT_MODEL = "tyto-l-16khz"          # family spec: newest Tyto L 16 kHz the SDK can load
FALLBACK_MODEL = "tyto-1.1-l-16khz"     # used only when the manifest cannot be read
WINDOW_SECONDS = 5  # fixed by the model family; informational only

# Dimensions are discovered from the SDK's AnalysisResult so a newer Tyto that adds or
# renames one shows up without a code change. This tuple only fixes display order for
# the names we know; unknown extras are appended alphabetically.
KNOWN_ORDER = (
    "risk_score",
    "noise",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "codec_degradation",
    "packet_loss",
)
HEADLINE = "risk_score"
# Neutral level meters: not degradations, excluded from the "why" argmax.
NEUTRAL_DIMS = frozenset({"speaker_loudness"})


def discover_dimensions(result_type=None) -> tuple[str, ...]:
    """Public attributes of the SDK's AnalysisResult, known names first."""
    result_type = result_type if result_type is not None else getattr(aic, "AnalysisResult", None)
    if not isinstance(result_type, type):
        return KNOWN_ORDER
    names = {n for n in dir(result_type) if not n.startswith("_")}
    if HEADLINE not in names:
        return KNOWN_ORDER
    ordered = [d for d in KNOWN_ORDER if d in names]
    ordered += sorted(names - set(KNOWN_ORDER))
    return tuple(ordered)


DIMENSIONS: tuple[str, ...] = discover_dimensions()
# Problem dimensions where higher = worse: everything except the headline and neutral meters.
DEGRADATION_DIMS: tuple[str, ...] = tuple(d for d in DIMENSIONS if d != HEADLINE and d not in NEUTRAL_DIMS)


def resolve_model_name(name: str) -> str:
    try:
        return resolve_or_fallback(name, FALLBACK_MODEL, kind=MODEL_KIND)
    except ModelResolutionError as e:
        sys.exit(f"{e}\nRun with --list-models to see what the installed SDK can load.")


def list_models() -> None:
    manifest = fetch_manifest()
    if manifest is None:
        sys.exit("Could not read the model manifest. Check network access to artifacts.ai-coustics.io.")
    version = compatible_version()
    print(format_catalog(manifest, version, kind=MODEL_KIND))
    try:
        default = resolve_model(DEFAULT_MODEL, manifest, version, kind=MODEL_KIND)
    except ModelResolutionError as e:
        default = f"nothing ({e})"
    print(f"\nSDK {aic.get_sdk_version()} loads artifact {version}. "
          f"Default spec '{DEFAULT_MODEL}' currently resolves to '{default}'.")
    print(f"AnalysisResult dimensions in this SDK: {', '.join(DIMENSIONS)}")


def build_analyzer(model_name: str, model_path: str | None, license_key: str) -> tuple[aic.FileAnalyzer, str]:
    if model_path:
        model = aic.Model.from_file(model_path)
        label = Path(model_path).stem
    else:
        artifact_id = resolve_model_name(model_name)
        print(f"model: {artifact_id} (resolved from '{model_name}'; SDK {aic.get_sdk_version()}, "
              f"artifact {compatible_version()})", file=sys.stderr)
        model = aic.Model.from_file(download_model(artifact_id))
        label = artifact_id
    try:
        return aic.FileAnalyzer(model, license_key), label
    except aic.ModelTypeUnsupportedError:
        sys.exit(
            f"'{label}' is not an analysis model, so the SDK's analyzer refuses it. Enhancement models "
            "belong to ai-coustics-speech-enhancement and VAD models to ai-coustics-voice-activity-detection. "
            "Run with --list-models to see analysis models."
        )


def analyze_file(
    analyzer: aic.FileAnalyzer, path: Path, step_seconds: float,
) -> tuple[list[dict], float]:
    """Return (per-window dicts, duration_seconds). Empty list if too short."""
    samples, sample_rate = load_mono_audio(path)
    step_samples = max(1, round(sample_rate * step_seconds))
    results = analyzer.analyze(samples, sample_rate, step_samples)
    rows = [{dim: float(getattr(r, dim)) for dim in DIMENSIONS} for r in results]
    return rows, len(samples) / sample_rate


# -- Aggregation --------------------------------------------------------------

def call_aggregates(rows: list[dict], threshold: float) -> dict:
    """Per-call summary: mean/p95/max per dimension, fraction of degraded windows,
    and the degradation dimension with the highest mean (the 'why')."""
    means = {d: float(np.mean([r[d] for r in rows])) for d in DIMENSIONS}
    p95 = {d: float(np.percentile([r[d] for r in rows], 95)) for d in DIMENSIONS}
    mx = {d: float(np.max([r[d] for r in rows])) for d in DIMENSIONS}
    risks = [r[HEADLINE] for r in rows]
    frac = sum(1 for v in risks if v >= threshold) / len(risks)
    worst_dim = max(DEGRADATION_DIMS, key=lambda d: means[d])
    return {
        "mean": means,
        "p95": p95,
        "max": mx,
        "frac_degraded": frac,
        "worst_dim": worst_dim,
    }


# Top of the Warn band; https://docs.ai-coustics.com/models/audio-insight/tyto
BAD_THRESHOLD = 0.50


def risk_band(risk: float, threshold: float = 0.30) -> str:
    if risk < threshold:
        return "🟢 Good"
    if risk <= BAD_THRESHOLD:
        return "🟡 Warn"
    return "🔴 Bad"


# -- Output builders ----------------------------------------------------------

def dashboard_entry(name: str, rows: list[dict], duration: float) -> dict:
    """Entry for the call-analysis.ai-coustics.com dashboard JSON."""
    return {
        "file": name,
        "duration_sec": round(duration, 2),
        "frames": {dim: [round(r[dim], 4) for r in rows] for dim in DIMENSIONS},
    }


def write_dashboard_json(out_json: Path, calls: list[dict]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"model": "Tyto", "calls": calls}, indent=2))


def write_means_csv(out_csv: Path, per_file: list[tuple[str, dict]]) -> None:
    """scores.csv-style file: filename index + tyto_<dim> mean columns."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [f"tyto_{d}" for d in DIMENSIONS]
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", *cols])  # pandas index_col=0 convention
        for name, agg in per_file:
            w.writerow([name, *(f"{agg['mean'][d]:.6f}" for d in DIMENSIONS)])


# -- LLM-judge context ----------------------------------------------------------

def degraded_intervals(rows: list[dict], step: float, threshold: float, duration: float) -> list[dict]:
    """Merge windows with risk >= threshold into intervals on the call timeline.

    Window i covers [i*step, i*step + WINDOW_SECONDS]. Degraded windows whose spans
    overlap in time are merged into one interval, so a 1 s hop over 5 s windows never
    yields overlapping intervals. Each interval reports its peak risk and the
    degradation dimension with the highest mean inside it (the driver).
    """
    intervals: list[dict] = []
    run: list[int] = []
    # Windows i and j overlap in time iff (j - i) * step < WINDOW_SECONDS.
    overlap_windows = max(1, math.ceil(WINDOW_SECONDS / step)) if step > 0 else 1

    def flush() -> None:
        if not run:
            return
        start = run[0] * step
        end = min(duration, run[-1] * step + WINDOW_SECONDS)
        sub = [rows[i] for i in run]
        driver = max(DEGRADATION_DIMS, key=lambda d: float(np.mean([r[d] for r in sub])))
        intervals.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "peak_risk": round(max(r[HEADLINE] for r in sub), 3),
            "driver": driver,
        })

    for i, r in enumerate(rows):
        if r[HEADLINE] < threshold:
            continue
        if run and i - run[-1] >= overlap_windows:   # gap wider than a window: new interval
            flush()
            run = []
        run.append(i)
    flush()
    return intervals


def _mmss(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def judge_context_markdown(name: str, rows: list[dict], duration: float, step: float,
                           threshold: float, model_label: str) -> str:
    """Compact block to paste into an LLM-judge prompt next to the call transcript."""
    agg = call_aggregates(rows, threshold)
    m = agg["mean"]
    intervals = degraded_intervals(rows, step, threshold, duration)
    band = risk_band(m[HEADLINE], threshold)
    lines = [
        f"### Audio quality context (ai-coustics Tyto, {model_label})",
        f"Call: {name} · {duration:.0f} s · {len(rows)} windows of {WINDOW_SECONDS} s",
        f"Overall risk_score: {m[HEADLINE]:.2f} ({band}) · {agg['frac_degraded']:.0%} of windows "
        f"degraded (risk ≥ {threshold}) · dominant driver: {agg['worst_dim']}",
    ]
    if intervals:
        lines.append(f"Degraded intervals (risk ≥ {threshold}):")
        for iv in intervals:
            lines.append(f"- {_mmss(iv['start'])}–{_mmss(iv['end'])} ({iv['end'] - iv['start']:.0f} s) "
                         f"peak risk {iv['peak_risk']:.2f} · driver: {iv['driver']}")
    else:
        lines.append(f"Degraded intervals: none (no window reached risk {threshold}).")
    lines += [
        "How to use this next to the transcript:",
        "- STT errors, repeated questions, misunderstandings or an unresponsive agent *inside* a "
        "degraded interval are likely caused by the incoming audio, not by the agent. Attribute "
        "them to audio before scoring the agent down on task completion or instruction following.",
        "- Outside degraded intervals the audio was clean; agent behaviour there can be judged at face value.",
        "- The driver says why: noise or speaker_reverb = environment; interfering_speech = another "
        "person talking (remedy: Quail Voice Focus isolates the primary speaker); codec_degradation "
        "or packet_loss = network/telephony path.",
        "- risk_score measures machine impact on VAD/STT/speech-to-speech, not how the call sounded "
        "to the caller. Do not use it as a tone, empathy or sentiment signal.",
    ]
    return "\n".join(lines) + "\n"


def write_judge_context(out_md: Path, blocks: list[str]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(blocks))


# -- Printing -----------------------------------------------------------------

def print_header(threshold: float, model_label: str) -> None:
    print(
        f"# Tyto ({model_label}) — {HEADLINE} + {len(DIMENSIONS) - 1} dimensions, all 0–1. "
        f"Higher = worse (neutral level meters: {', '.join(sorted(NEUTRAL_DIMS & set(DIMENSIONS))) or 'none'}).\n"
        f"# {WINDOW_SECONDS}-s windows; band 🟢<{threshold} 🟡{threshold}–{BAD_THRESHOLD} "
        f"🔴>{BAD_THRESHOLD}; 'why' = degradation dimension with the highest mean."
    )


def _fmt_dims(d: dict) -> str:
    return "  ".join(f"{k}={d[k]:.3f}" for k in DIMENSIONS)


def print_single_file(name: str, rows: list[dict], duration: float, step: float,
                      threshold: float, model_label: str = "") -> None:
    print_header(threshold, model_label)
    print(f"\n{name}: {len(rows)} window(s), {duration:.1f}s audio, {step:g}s hop")
    for i, r in enumerate(rows):
        start = i * step
        print(f"  win {i:>3} [{start:>5.1f}s]  {_fmt_dims(r)}")

    agg = call_aggregates(rows, threshold)
    m = agg["mean"]
    print("  " + "-" * 70)
    print(f"  mean    {_fmt_dims(m)}")
    print(
        f"  risk    mean={m[HEADLINE]:.3f}  p95={agg['p95'][HEADLINE]:.3f}  "
        f"max={agg['max'][HEADLINE]:.3f}  "
        f"frac≥{threshold}={agg['frac_degraded']:.0%}  "
        f"→ {risk_band(m[HEADLINE], threshold)}"
    )
    print(f"  why     worst degradation dimension: {agg['worst_dim']} "
          f"(mean={m[agg['worst_dim']]:.3f})")


def print_folder(
    files: list[Path], step: float, threshold: float,
    analyzer: aic.FileAnalyzer, model_label: str = "",
) -> tuple[list[tuple[str, dict]], list[dict], list[str], list[str]]:
    """Score every file; return (per-file aggregates, dashboard entries, judge blocks, failed names).

    A file that cannot be read or scored is reported on its own line and skipped, so one
    bad recording does not lose the results of the rest of the batch.
    """
    print_header(threshold, model_label)
    per_file: list[tuple[str, dict]] = []
    calls: list[dict] = []
    judge_blocks: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    name_w = max(len(f.name) for f in files)
    for i, f in enumerate(files, 1):
        try:
            rows, duration = analyze_file(analyzer, f, step)
        except Exception as e:  # unreadable file, SDK error: report and carry on
            failed.append(f.name)
            print(f"[{i:>3}/{len(files)}] {f.name:<{name_w}}  ERROR: {e}")
            continue
        if not rows:
            skipped.append(f.name)
            print(f"[{i:>3}/{len(files)}] {f.name:<{name_w}}  (skipped: < one {WINDOW_SECONDS}s window)")
            continue
        agg = call_aggregates(rows, threshold)
        per_file.append((f.name, agg))
        calls.append(dashboard_entry(f.name, rows, duration))
        judge_blocks.append(judge_context_markdown(f.name, rows, duration, step, threshold, model_label))
        m = agg["mean"]
        print(
            f"[{i:>3}/{len(files)}] {f.name:<{name_w}}  "
            f"risk={m[HEADLINE]:.3f} {risk_band(m[HEADLINE], threshold).split()[0]}  "
            f"frac≥{threshold}={agg['frac_degraded']:.0%}  why={agg['worst_dim']}"
        )

    if not per_file:
        sys.exit("no files produced a usable score (all shorter than one window or unreadable)")

    print(f"\n=== Summary across {len(per_file)} files ===")
    if skipped:
        print(f"(skipped {len(skipped)}: {', '.join(skipped[:3])}{'…' if len(skipped) > 3 else ''})")
    if failed:
        print(f"(failed {len(failed)}: {', '.join(failed[:3])}{'…' if len(failed) > 3 else ''})")
    header = f"{'dimension':<20} {'mean':>10} {'stdev':>10} {'min':>10} {'max':>10}"
    print(header)
    print("-" * len(header))
    for d in DIMENSIONS:
        col = [agg["mean"][d] for _, agg in per_file]
        stdev = statistics.stdev(col) if len(col) > 1 else 0.0
        print(f"{d:<20} {statistics.mean(col):>10.4f} {stdev:>10.4f} {min(col):>10.4f} {max(col):>10.4f}")
    return per_file, calls, judge_blocks, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("audio_path", nargs="?", help="Audio file or folder of audio files")
    ap.add_argument("--step-seconds", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--model-name", default=DEFAULT_MODEL,
                    help=f"Family spec or exact id (default: {DEFAULT_MODEL} = newest Tyto L 16 kHz)")
    ap.add_argument("--model-path", default=None, help="Local .aicmodel (overrides --model-name)")
    ap.add_argument("--list-models", action="store_true", help="List loadable analysis models and exit")
    ap.add_argument("--license-key", default=None)
    ap.add_argument("--out-json", type=Path, default=None, help="Write dashboard JSON")
    ap.add_argument("--out-csv", type=Path, default=None, help="Folder mode: per-file means CSV")
    ap.add_argument("--out-judge", type=Path, default=None,
                    help="Write Markdown audio-quality context for an LLM judge (one block per call)")
    args = ap.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.audio_path:
        ap.error("audio_path is required (or use --list-models)")

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        sys.exit(f"audio path does not exist: {audio_path}")
    if args.out_csv is not None and not audio_path.is_dir():
        sys.exit("--out-csv only applies to folder mode")
    if audio_path.is_file() and audio_path.suffix.lower() not in SUPPORTED_EXTS:
        sys.exit(f"Unsupported format: {audio_path.suffix} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")

    license_key = resolve_license(args.license_key)
    analyzer, model_label = build_analyzer(args.model_name, args.model_path, license_key)

    if audio_path.is_file():
        rows, duration = analyze_file(analyzer, audio_path, args.step_seconds)
        if not rows:
            sys.exit(f"{audio_path.name}: shorter than one {WINDOW_SECONDS}s window — nothing to score")
        print_single_file(audio_path.name, rows, duration, args.step_seconds, args.threshold, model_label)
        if args.out_json is not None:
            write_dashboard_json(args.out_json, [dashboard_entry(audio_path.name, rows, duration)])
            print(f"\nwrote dashboard JSON to {args.out_json}")
        if args.out_judge is not None:
            write_judge_context(args.out_judge, [judge_context_markdown(
                audio_path.name, rows, duration, args.step_seconds, args.threshold, model_label)])
            print(f"wrote LLM-judge context to {args.out_judge}")
        return

    files = sorted(p for p in audio_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        sys.exit(f"no audio files in {audio_path} (looked for {sorted(SUPPORTED_EXTS)})")
    per_file, calls, judge_blocks, failed = print_folder(files, args.step_seconds, args.threshold, analyzer, model_label)

    if args.out_csv is not None:
        write_means_csv(args.out_csv, per_file)
        print(f"\nwrote per-file means to {args.out_csv}")
    if args.out_json is not None:
        write_dashboard_json(args.out_json, calls)
        print(f"wrote dashboard JSON ({len(calls)} calls) to {args.out_json}")
        print("upload it at https://call-analysis.ai-coustics.com/")
    if args.out_judge is not None:
        write_judge_context(args.out_judge, judge_blocks)
        print(f"wrote LLM-judge context ({len(judge_blocks)} calls) to {args.out_judge}")
    if failed:
        sys.exit(f"{len(failed)} of {len(files)} files could not be scored (see ERROR lines); "
                 "outputs cover the rest.")


if __name__ == "__main__":
    main()
