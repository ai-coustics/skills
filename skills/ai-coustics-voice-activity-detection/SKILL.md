---
name: ai-coustics-voice-activity-detection
description: >
  Detect speech in audio files (single file or folder) with the ai-coustics VAD models via
  the AIC SDK — per-block speech probabilities, post-processed speech/no-speech decisions,
  and speech segments with delay-compensated timestamps. VAD Multi Speaker (ms) fires on
  any audible speech; VAD Voice Focus (vf) fires on the primary speaker only. Always uses
  the newest VAD build the installed SDK can load. Trigger this skill when the user asks
  to: run voice activity detection, detect speech or silence in a recording, find speech
  segments or speech onsets, measure how much of a file is speech, tune VAD sensitivity
  for turn-taking or endpointing, or compare the ai-coustics VAD with Silero or WebRTC
  VAD. Also trigger when the user references "VAD", "voice activity", "speech detection",
  "vad-ms", "vad-vf", "endpointing", or "turn-taking".
license: MIT
metadata:
  author: ai-coustics
  version: "2.0.0"
---

# Voice Activity Detection — ai-coustics VAD via the AIC SDK

Runs one of the ai-coustics VAD models over a single audio file or a folder and reports,
per file, the speech segments plus the raw per-block probabilities behind them. The
audio is never modified. Models **auto-download on first run** (cached to
`~/.cache/aic-models`) and run on CPU.

| Product | Alias | Family spec | Detects |
|---|---|---|---|
| **VAD Multi Speaker** (default) | `ms` | `vad-ms-16khz` | Any audible speech, any speaker. Robust to noise, music, far-field. |
| **VAD Voice Focus** | `vf` | `vad-vf-16khz` | The primary speaker only; ignores interfering and background speech. |

**Nothing here is pinned to an SDK or model generation.** `ms` and `vf` resolve at run
time to the newest build of each family the installed SDK can load. Block length,
prediction delay and the parameter defaults are all read from the model, so a new VAD
version with a different window or different defaults works without a change here.

**Note:** `{BASE_DIR}` = this skill's directory. Claude Code prints it as the base
directory when the skill loads; in other agents it is wherever the skill was installed.

---

## Prerequisites

- `uv` on PATH (the script declares its own dependencies inline and runs with `uv run`).
- An AIC SDK license key available via one of `AIC_SDK_LICENSE_KEY`,
  `AIC_LICENSE_KEY`, `AIC_SDK_LICENSE` — or passed with `--license-key`.
  Generate a self-service key at https://developers.ai-coustics.com.
- Network access to `artifacts.ai-coustics.io` for the model catalog and download.
  Offline: pass a pre-downloaded `.aicmodel` with `--model-path`.

If no key is present, ask the user to set the env var before running. Never echo the
key value into chat or logs.

---

## Steps

1. **Confirm inputs.** A single file (`.wav`/`.flac`/`.mp3`/`.ogg`) or a folder.
   Pick `ms` unless the user wants only the primary speaker, then `vf`.

2. **Run detect.py:**
   ```bash
   uv run {BASE_DIR}/scripts/detect.py <audio_path>
   ```
   - Add `--out-csv segments.csv` for a segment table (file, start, end, duration).
   - Add `--out-json vad.json` for per-block probabilities and decisions plus segments.
   - Add `--print-blocks` (single file) to see every block's probability.
   - Tune with `--sensitivity`, `--speech-hold`, `--min-speech`. Defaults come from the
     model file and are printed in the header as "parameters in effect".
   The script prints the resolved model on stderr, e.g.
   `model: vad-ms-2.1-xxs-16khz (resolved from 'ms'; SDK 0.24.0, artifact v7)`. Relay it.

3. **Explain the output.** Lead with speech coverage (seconds and percent), segment
   count and first onset. If the user is tuning for turn-taking, point at the
   `probability` arrays in the JSON: the docs recommend choosing a threshold from the
   probability distribution on real audio, then setting `--sensitivity`.

---

## How the decision is made

The model outputs a **speech probability** per block. The SDK compares it with
**sensitivity** and smooths the result with two durations. All three defaults are read
from the model file, so treat the header's "parameters in effect" as the starting point.

| Parameter | Flag | Meaning |
|---|---|---|
| Sensitivity | `--sensitivity` | Probability threshold, 0.0–1.0. **Higher = fewer detections.** |
| Speech hold duration | `--speech-hold` | Seconds speech stays "on" after it stops. Stabilises on→off. |
| Minimum speech duration | `--min-speech` | Seconds speech must persist before it counts. Stabilises off→on. |

Durations are rounded to the model's block length, so the value in effect can differ
from the one you passed. If the model's default minimum speech duration is 0 s you may
see brief micro-segments on breaths and clicks; `--min-speech 0.1` removes them for
offline segmentation.

**Timestamps are compensated** for the VAD's prediction delay, read from the SDK per
run, so segment starts and ends line up with the input audio.

**To see what is loadable right now:**
```bash
uv run {BASE_DIR}/scripts/detect.py --list-models
```
Do not quote model versions from memory; they change.

---

## detect.py arguments

| Argument | Required | Description |
|---|---|---|
| `audio_path` | Yes | Single audio file OR a folder of audio files |
| `--model-name NAME` | No | `ms`, `vf`, a family spec, or an exact VAD artifact id (default `ms`) |
| `--model-path PATH` | No | Local `.aicmodel` file (takes precedence over `--model-name`) |
| `--list-models` | No | Print loadable VAD models and what `ms` / `vf` resolve to, then exit |
| `--sensitivity F` | No | Probability threshold 0.0–1.0 (default: model) |
| `--speech-hold F` | No | Speech hold duration in seconds (default: model) |
| `--min-speech F` | No | Minimum speech duration in seconds (default: model) |
| `--print-blocks` | No | Single-file mode: one line per block with probability and decision |
| `--license-key KEY` | No | Override the license env vars |
| `--out-json PATH` | No | Per-file blocks (`probability`, `speech`), segments, parameters in effect |
| `--out-csv PATH` | No | Segment table: `file, segment, start_sec, end_sec, duration_sec` |

---

## Output

Single file (values illustrative):

```
model: vad-ms-2.1-xxs-16khz (resolved from 'ms'; SDK 0.24.0, artifact v7)
# ai-coustics VAD (vad-ms-2.1-xxs-16khz); parameters in effect: sensitivity=0.600  speech_hold_duration=0.100  minimum_speech_duration=0.000
# 15 ms blocks, prediction delay 30 ms (timestamps already compensated); audio is not modified.

harvard.wav: 33.62s @ 8000 Hz, 2242 blocks

    #     start       end      dur
    1      0.48      3.27     2.79
  ...
  speech  23.62s of 33.62s (70%)  segments=14  first_onset=0.48s  longest=2.79s  mean_p=0.688
```

Folder mode prints one line per file (speech %, seconds, segments, onset, mean
probability) and a totals line, and names files with no speech at all. A file that
cannot be read is reported as `ERROR <file>: <reason>` and skipped; the run then exits
non-zero after writing the outputs for the rest.

---

## Key implementation details

- **Model resolution** lives in `scripts/_aic_models.py`: fetch the manifest, filter to
  the artifact version from `aic.get_compatible_model_version()`, pick the newest build
  of the requested family. Falls back to the last known id if the manifest is
  unreachable and says so on stderr.
- **Dedicated `aic.Vad`.** `aic.Vad(model, license_key, aic.ProcessorConfig.optimal(model,
  sample_rate=fs))`, fed mono float32 blocks of `config.block_size`; after each block
  the script reads `raw_vad_probability()` and `is_speech_detected()`. The last block is
  zero-padded.
- **Segments** are contiguous runs of `is_speech_detected() == True`, shifted back by
  `get_prediction_delay()` samples and clipped to the file duration.
- **One `Vad` per file** so folders with mixed sample rates work; the model is loaded once.
- **Type enforcement is the SDK's.** Passing an enhancement or Tyto model fails at
  `aic.Vad` creation; the script explains and names the right skill.
- **Real-time.** For live streams use `aic.Vad` (or `aic.VadAsync`) beside your
  enhancement processor on the **same original input block**, never on the enhanced
  output. This skill covers offline files only.

## Troubleshooting

- `No AIC SDK license key` → set one of the env vars in Prerequisites, or pass `--license-key`.
- `'…' is an enhancement model, but this skill drives vad models` → use
  `ai-coustics-speech-enhancement` (or `ai-coustics-audio-insight` for Tyto).
- `No model matching '…' is published for the installed SDK` → retired or misspelled id;
  the message lists what is available. Prefer `ms` / `vf`.
- `out of range for this model` → sensitivity must be 0–1; durations have model-specific
  upper bounds.
- Many tiny segments → raise `--min-speech` (e.g. `0.1`) or `--sensitivity`.
- Sandboxed agents without network access cannot download the model. Pass a
  pre-downloaded `.aicmodel` with `--model-path`.

## Reference

- VAD product page: https://docs.ai-coustics.com/models/voice-activity-detection/vad
- Model specs: https://docs.ai-coustics.com/reference/sdk/models
- Sensitivity semantics and the dedicated-VAD migration: https://docs.ai-coustics.com/reference/deprecated/energy-vad-to-dedicated-vad
- Docs index: https://docs.ai-coustics.com/llms.txt
