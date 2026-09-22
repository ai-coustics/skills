---
name: ai-coustics-speech-enhancement
description: >
  Enhance audio files (single file or entire folder) with ai-coustics speech-enhancement
  models — Quail Voice Focus, Quail Multi Speaker, Rook Multi Speaker — via the AIC SDK,
  with async parallel processing. Always uses the newest model build the installed SDK
  can load. Trigger this skill when the user asks to: enhance audio with ai-coustics or the
  AIC SDK, clean up speech recordings, improve speech quality, apply voice focus, isolate the primary
  speaker, remove background noise from a call or a folder of files, or run enhance.py.
  Also trigger when the user references "quail", "rook", "voice focus", "multi speaker",
  "sdk-enhance", or asks to batch-enhance audio.
license: MIT
metadata:
  author: ai-coustics
  version: "2.0.0"
---

# Speech Enhancement — ai-coustics models via the AIC SDK

Runs AIC SDK speech enhancement on a single audio file or a recursive folder of files
with an async worker pool. Models **auto-download on first run** (cached to
`~/.cache/aic-models`) and run on CPU.

`uv` installs the newest `aic-sdk`, the script asks that SDK which artifact version it loads, reads the public
model catalog at run time, and resolves the requested model family to the newest build
available. When ai-coustics ships a new Voice Focus version, the default picks it up
without a change to this skill.

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

1. **Confirm inputs.** If the model is not specified, use the default (`vf`, newest
   Quail Voice Focus L at 16 kHz) and enhancement level `1.0`. If no output path is
   given, use `{input}_enhanced` for folders or `{stem}_enhanced.{ext}` for single files.
   If the user asks which models exist, run `--list-models` rather than answering from
   memory.

2. **Run enhance.py:**
   ```bash
   uv run {BASE_DIR}/scripts/enhance.py <input> <output> --model-name vf --enhancement-level 1.0
   ```
   For folders, set `--workers` to the number of files (cap at 16). The script prints
   the concrete artifact it resolved to on stderr, e.g.
   `model: quail-vf-2.2-l-16khz (resolved from 'vf'; SDK 0.24.0, artifact v7)`. Relay
   that line so the user knows which build processed their audio.

3. **Verify** — confirm output file(s) exist and report path, duration, sample rate,
   and the resolved model id.

---

## Choosing a model

Three product lines, all documented at https://docs.ai-coustics.com/reference/sdk/models:

| Product | Alias | Family spec | Use it for |
|---|---|---|---|
| **Quail Voice Focus** | `vf` (default), `vf s`, `vf 48k` | `quail-vf-l-16khz`, `quail-vf-s-16khz`, `quail-vf-l-48khz` | Isolate the primary speaker for Voice AI / STT. Suppresses interfering speakers as well as noise. |
| **Quail Multi Speaker** | `ms`, `quail ms s`, `quail ms 8k` | `quail-ms-l-16khz`, `quail-ms-s-16khz`, `quail-ms-l-8khz` | Enhance all speakers for Voice AI / STT (meetings, multi-party calls). |
| **Rook Multi Speaker** | `rook`, `rook s`, `rook 48k` | `rook-ms-l-16khz`, `rook-ms-s-16khz`, `rook-ms-l-48khz` | Perceptual enhancement for human listeners (natural sound, 48 kHz builds). |

`ms` = Multi Speaker, `vf` = Voice Focus. `l` / `s` are model sizes (large is more
accurate, small is faster). The trailing rate is the model's native sample rate; the SDK
resamples other input rates internally.

**How names resolve.** A *family spec* is a model id with the version left out
(`quail-vf-l-16khz`). It resolves to the newest version of that family the installed SDK
can load; leaving out the size picks the largest. An exact id (`quail-vf-2.2-l-16khz`)
is used as-is when loadable, otherwise the script exits and lists what is. Pre-rename
ids (`quail-l-16khz`, `rook-l-48khz`, …) map onto the current products automatically.

**To see what is loadable right now:**
```bash
uv run {BASE_DIR}/scripts/enhance.py --list-models
```
This is the source of truth for model availability. Do not quote model versions from
memory; they change.

---

## enhance.py arguments

| Argument | Required | Description |
|---|---|---|
| `input` | Yes | `.wav`/`.flac`/`.mp3`/`.ogg` file or folder |
| `output` | Yes | Output file or folder (mirrors input directory structure) |
| `--model-name NAME` | No | Alias, family spec, or exact artifact id (default: `quail-vf-l-16khz`) |
| `--model-path PATH` | No | Local `.aicmodel` file (takes precedence over `--model-name`) |
| `--list-models` | No | Print loadable enhancement models and the current default resolution, then exit |
| `--enhancement-level F` | No | 0.0–1.0, default `1.0` (ignored by models with a fixed level) |
| `--workers N` | No | Parallel processors for folder mode, default `8` |
| `--license-key KEY` | No | Override license key env var |

---

## Output

Single-file mode prints the resolved model on stderr and one line on stdout:

```
model: quail-vf-2.2-l-16khz (resolved from 'vf'; SDK 0.24.0, artifact v7)
Saved: call_enhanced.wav  (42.10s, 16000Hz)
```

Folder mode shows a progress bar, logs `ERROR <file>: <reason>` for any file that
fails without stopping the batch, and ends with `Done. Output: <folder>`. If any file
failed it ends with `Done with errors: N of M files failed` and a non-zero exit code.

---

## Key implementation details

- **Model resolution** lives in `scripts/_aic_models.py`: fetch the manifest, filter to
  the artifact version from `aic.get_compatible_model_version()`, rank by version then
  size. If the manifest is unreachable it falls back to the last id known to work and
  says so on stderr.
- **Queue + N workers**: each worker owns one `ProcessorAsync`; picks the next file the
  moment it finishes — no batch barrier.
- **Mono processing**: multi-channel input is downmixed before processing; output is
  mono at the input's sample rate.
- **Latency compensation**: the processor's output lags its input by
  `get_audio_delay()` samples, so the script appends that many zeros to the input,
  processes, and drops the same number from the output head; the result is aligned with
  the input and has exactly its length.
- **Type enforcement is the SDK's.** Passing a VAD or Tyto model fails at processor
  creation with `ModelTypeUnsupportedError`; the script explains and names the right skill.
- **Recursive**: folder mode walks subdirectories and mirrors structure under `output/`.

## Troubleshooting

- `No AIC SDK license key` → set one of the env vars in Prerequisites, or pass `--license-key`.
- `No model matching '…' is published for the installed SDK` → the id is retired for
  this SDK generation or misspelled. The message lists what is available; prefer a
  family spec or alias so this cannot recur.
- `'…' is a vad model, but this skill drives enhancement models` → use the
  `ai-coustics-voice-activity-detection` skill (or `ai-coustics-audio-insight` for Tyto).
- `note: could not read …/manifest.json` → catalog unreachable; the script continues
  with the last known id of the requested family, or stops if it has none for that
  family (pass an exact id or `--model-path`). Check network access if a newer model
  was expected.
- Sandboxed agents without network access cannot download models. Pass a
  pre-downloaded `.aicmodel` with `--model-path`.

## Reference

- `references/sdk_api.md` — AIC SDK Python API cheat-sheet (model loading, async
  processor, latency compensation, model resolution).
- Models: https://docs.ai-coustics.com/reference/sdk/models
- Model renames: https://docs.ai-coustics.com/models/older-models/model-naming-changes
- Catalog (machine-readable): https://artifacts.ai-coustics.io/manifest.json
- Docs index: https://docs.ai-coustics.com/llms.txt
