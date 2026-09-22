# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "aic-sdk>=3.1.0",
#   "certifi",
#   "numpy",
#   "soundfile",
#   "tqdm",
# ]
# ///
"""
Enhance audio files with ai-coustics speech-enhancement models via the AIC SDK.

The model is resolved at run time against the public artifact manifest and the
installed SDK's compatible artifact version, so the default always points at the
newest Quail Voice Focus build the SDK can load. No SDK or model generation is pinned.

Note: the SDK processes mono. Multi-channel input is downmixed before processing, so
output files are mono at the input's sample rate.

Usage (single file):
  uv run enhance.py input.wav output.wav [options]

Usage (folder):
  uv run enhance.py /path/to/input/ /path/to/output/ [options]

Options:
  --model-name NAME       Alias ('vf', 'vf s', 'ms', 'rook', ...), family spec
                          ('quail-vf-l-16khz' = newest Voice Focus L 16 kHz), or exact
                          artifact id (default: newest quail-vf-l-16khz)
  --model-path PATH       Local .aicmodel file (takes precedence over --model-name)
  --list-models           Print the enhancement models the installed SDK can load, then exit
  --enhancement-level F   0.0-1.0 (default: 1.0)
  --workers N             Parallel processors for folder mode (default: 8)
  --license-key KEY       Override AIC_SDK_LICENSE_KEY / AIC_LICENSE_KEY / AIC_SDK_LICENSE
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import aic_sdk as aic
import numpy as np
import soundfile as sf
from tqdm import tqdm

from _aic_models import (
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

MODEL_KIND = "enhancement"

# Family specs: a model id without the version. The helper resolves each to the
# newest version the installed SDK can load, largest size when the size is omitted.
DEFAULT_MODEL = "quail-vf-l-16khz"
# Used only when the manifest cannot be read; the last id known to work.
FALLBACK_MODEL = "quail-vf-2.2-l-16khz"

# Friendly aliases -> family spec. Product names follow docs.ai-coustics.com:
# Quail Voice Focus (quail-vf), Quail Multi Speaker (quail-ms), Rook Multi Speaker (rook-ms).
ALIASES: dict[str, str] = {
    "vf": "quail-vf-l-16khz",
    "voice focus": "quail-vf-l-16khz",
    "quail vf": "quail-vf-l-16khz",
    "vf l": "quail-vf-l-16khz",
    "vf s": "quail-vf-s-16khz",
    "vf 48k": "quail-vf-l-48khz",
    "vf l 48k": "quail-vf-l-48khz",
    "ms": "quail-ms-l-16khz",
    "multi speaker": "quail-ms-l-16khz",
    "quail ms": "quail-ms-l-16khz",
    "quail": "quail-ms-l-16khz",
    "quail ms s": "quail-ms-s-16khz",
    "quail ms 8k": "quail-ms-l-8khz",
    "rook": "rook-ms-l-16khz",
    "rook ms": "rook-ms-l-16khz",
    "rook l": "rook-ms-l-16khz",
    "rook s": "rook-ms-s-16khz",
    "rook 48k": "rook-ms-l-48khz",
    "rook l 48k": "rook-ms-l-48khz",
    "rook s 48k": "rook-ms-s-48khz",
    "rook 8k": "rook-ms-l-8khz",
    # Pre-rename ids still map onto the current products (see docs: Model Naming Changes).
    "quail-l-16khz": "quail-ms-l-16khz",
    "quail-l-8khz": "quail-ms-l-8khz",
    "quail-s-16khz": "quail-ms-s-16khz",
    "quail-s-8khz": "quail-ms-s-8khz",
    "rook-l-16khz": "rook-ms-l-16khz",
    "rook-l-8khz": "rook-ms-l-8khz",
    "rook-l-48khz": "rook-ms-l-48khz",
    "rook-s-16khz": "rook-ms-s-16khz",
    "rook-s-8khz": "rook-ms-s-8khz",
    "rook-s-48khz": "rook-ms-s-48khz",
}


def alias_to_spec(name: str) -> str:
    """Map a friendly alias to a family spec; anything else passes through unchanged."""
    return ALIASES.get(name.strip().lower(), name.strip())


def resolve_model_name(name: str) -> str:
    """Alias or spec -> artifact id the installed SDK can load."""
    try:
        return resolve_or_fallback(alias_to_spec(name), FALLBACK_MODEL, kind=MODEL_KIND)
    except ModelResolutionError as e:
        sys.exit(f"{e}\nRun with --list-models to see what the installed SDK can load.")


def list_models() -> None:
    manifest = fetch_manifest()
    if manifest is None:
        sys.exit("Could not read the model manifest. Check network access to artifacts.ai-coustics.io.")
    version = compatible_version()
    legacy_ids = {k for k in ALIASES if "-" in k}  # pre-rename ids listed in ALIASES
    print(format_catalog(manifest, version, kind=MODEL_KIND, hide=legacy_ids))
    try:
        default = resolve_model(DEFAULT_MODEL, manifest, version, kind=MODEL_KIND)
    except ModelResolutionError as e:
        default = f"nothing ({e})"
    print(f"\nSDK {aic.get_sdk_version()} loads artifact {version}. "
          f"Default spec '{DEFAULT_MODEL}' currently resolves to '{default}'.")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

async def _process_chunk(processor: aic.ProcessorAsync, chunk: np.ndarray,
                          buffer_size: int) -> np.ndarray:
    valid = chunk.shape[0]
    if valid < buffer_size:  # only the final chunk needs zero-padding to a full block
        buf = np.zeros(buffer_size, dtype=np.float32)
        buf[:valid] = chunk
        chunk = buf
    out = await processor.process_async(chunk)
    return out[:valid]


async def _process_file(in_path: str, out_path: str, model: aic.Model,
                        processor: aic.ProcessorAsync, enhancement_level: float) -> None:
    audio, fs = load_mono_audio(in_path)

    config = aic.ProcessorConfig.optimal(model, sample_rate=fs)
    await processor.initialize_async(config)
    ctx = processor.get_context()
    ctx.reset()

    try:
        ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, enhancement_level)
    except aic.ParameterFixedError:
        pass  # model has a fixed enhancement level

    # The processor's output lags its input by `latency` samples. Append that many
    # zeros so the tail of the audio is flushed out, then drop the first `latency`
    # output samples: the result is aligned with the input and has the same length.
    latency = ctx.get_audio_delay()
    padded = np.concatenate([audio, np.zeros(latency, dtype=np.float32)])

    output = np.zeros_like(padded)
    block_size = config.block_size
    for start in range(0, padded.shape[0], block_size):
        chunk = padded[start:start + block_size]
        processed = await _process_chunk(processor, chunk, block_size)
        output[start:start + processed.shape[0]] = processed

    output = output[latency:]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, output, fs)


def make_processor(model: aic.Model, license_key: str) -> aic.ProcessorAsync:
    """Create the async processor; explain clearly if the model is not an enhancement model."""
    try:
        return aic.ProcessorAsync(model, license_key)
    except aic.ModelTypeUnsupportedError:
        sys.exit(
            f"'{model.get_id()}' is not an enhancement model, so the SDK's processor refuses it. "
            "VAD models belong to the ai-coustics-voice-activity-detection skill and Tyto to "
            "ai-coustics-audio-insight. Run with --list-models to see enhancement models."
        )


# ---------------------------------------------------------------------------
# Folder mode: asyncio Queue + N worker coroutines
# ---------------------------------------------------------------------------

async def _enhance_folder_async(file_pairs: list[tuple[str, str]], model: aic.Model,
                                  license_key: str, enhancement_level: float,
                                  n_workers: int) -> list[str]:
    """Enhance every pair; return the input paths that failed (the batch never stops)."""
    queue: asyncio.Queue = asyncio.Queue()
    for pair in file_pairs:
        await queue.put(pair)
    for _ in range(n_workers):
        await queue.put(None)  # sentinels
    failed: list[str] = []

    with tqdm(total=len(file_pairs), desc="Enhancing", unit="file") as pbar:
        async def worker():
            proc = make_processor(model, license_key)
            while True:
                item = await queue.get()
                if item is None:
                    return
                in_path, out_path = item
                try:
                    await _process_file(in_path, out_path, model, proc, enhancement_level)
                except Exception as e:
                    tqdm.write(f"ERROR {in_path}: {e}")
                    failed.append(in_path)
                finally:
                    pbar.update(1)

        tasks = [asyncio.create_task(worker()) for _ in range(n_workers)]
        await asyncio.gather(*tasks)
    return failed


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: str | None, model_name: str) -> aic.Model:
    if model_path:
        if not os.path.exists(model_path):
            sys.exit(f"Model file not found: {model_path}")
        return aic.Model.from_file(model_path)
    artifact_id = resolve_model_name(model_name)
    print(f"model: {artifact_id} (resolved from '{model_name}'; SDK {aic.get_sdk_version()}, "
          f"artifact {compatible_version()})", file=sys.stderr)
    return aic.Model.from_file(download_model(artifact_id))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enhance audio with the AIC SDK")
    parser.add_argument("input",  nargs="?", help="Input .wav/.flac/.mp3/.ogg file or folder")
    parser.add_argument("output", nargs="?", help="Output file or folder")
    parser.add_argument("--model-path",        default=None,         help="Local .aicmodel file")
    parser.add_argument("--model-name",        default=DEFAULT_MODEL,
                        help="Alias ('vf', 'ms', 'rook'), family spec ('quail-vf-l-16khz') or exact id "
                             f"(default: {DEFAULT_MODEL} = newest Voice Focus L 16 kHz)")
    parser.add_argument("--list-models",       action="store_true", help="List loadable enhancement models and exit")
    parser.add_argument("--enhancement-level", type=float, default=1.0, help="EL 0.0-1.0 (default 1.0)")
    parser.add_argument("--workers",           type=int,   default=8,   help="Parallel processors (folder mode)")
    parser.add_argument("--license-key",       default=None, help="Override AIC_SDK_LICENSE_KEY / AIC_LICENSE_KEY / AIC_SDK_LICENSE")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return
    if not args.input or not args.output:
        parser.error("input and output are required (or use --list-models)")

    license_key = resolve_license(args.license_key)
    inp = Path(args.input)
    out = Path(args.output)
    if not inp.exists():
        sys.exit(f"Input not found: {inp}")
    if inp.is_file() and inp.suffix.lower() not in SUPPORTED_EXTS:
        sys.exit(f"Unsupported format: {inp.suffix} (supported: {', '.join(sorted(SUPPORTED_EXTS))})")
    out_path = str(out) if (inp.is_dir() or out.suffix) else str(out / inp.name)
    if Path(out_path).resolve() == inp.resolve():
        sys.exit(f"Output would overwrite the input: {inp}. Choose a different output path.")

    model = load_model(args.model_path, args.model_name)

    if inp.is_file():
        asyncio.run(_process_file(str(inp), out_path, model,
                                  make_processor(model, license_key),
                                  args.enhancement_level))
        info = sf.info(out_path)
        print(f"Saved: {out_path}  ({info.duration:.2f}s, {info.samplerate}Hz)")

    else:
        file_pairs = []
        for root, _, files in os.walk(inp):
            for f in files:
                if Path(f).suffix.lower() in SUPPORTED_EXTS:
                    in_path = os.path.join(root, f)
                    rel = os.path.relpath(in_path, inp)
                    file_pairs.append((in_path, str(out / rel)))

        if not file_pairs:
            sys.exit(f"No supported audio files found under {inp}")

        n_workers = max(1, min(args.workers, len(file_pairs)))
        print(f"Enhancing {len(file_pairs)} files with {n_workers} workers ...")
        failed = asyncio.run(_enhance_folder_async(file_pairs, model, license_key,
                                                    args.enhancement_level, n_workers))
        if failed:
            sys.exit(f"Done with errors: {len(failed)} of {len(file_pairs)} files failed (see ERROR lines). "
                     f"Output for the rest: {out}")
        print(f"Done. Output: {out}")


if __name__ == "__main__":
    main()
