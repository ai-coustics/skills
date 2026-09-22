# Skill card: ai-coustics-speech-enhancement

**Description:** Enhance speech recordings (single file or folder) with the newest ai-coustics Quail Voice Focus, Quail Multi Speaker or Rook Multi Speaker build the installed SDK can load, via the AIC SDK, with async parallel processing.

**Owner:** ai-coustics · **License:** MIT (skill) · the AIC SDK and model artifacts are governed by the [ai-coustics terms](https://ai-coustics.com/terms).

## Use case
Developers who need to enhance speech, reduce reverberation or isolate a speaker in recordings before transcription, analysis, or playback, and want to batch-process folders reproducibly.

## Requirements / dependencies
- **Requires credential:** yes, an AIC SDK license key via `AIC_SDK_LICENSE_KEY`, `AIC_LICENSE_KEY` or `AIC_SDK_LICENSE`. Self-service keys at https://developers.ai-coustics.com.
- **Network:** reads the model catalog and downloads the model artifact on first run to `~/.cache/aic-models`. Offline use is possible with `--model-path`.
- **Runtime:** `uv`, Python ≥ 3.11, `aic-sdk>=3.1.0`, CPU only.

Do not paste the license key into prompts, logs or output. Use least-privilege keys and rotate as appropriate.

## Skill output
- **Types:** enhanced audio files (same container as input, mono, input sample rate), progress on stdout.
- **Side effects:** writes to the output path the user names; creates the model cache directory on first run. Never overwrites the input.

## Known risks and mitigations
- **Mono output:** SDK 3.x is mono-only; multi-channel input is downmixed. The skill states this up front.
- **Model drift:** models are resolved against the live catalog for the installed SDK, so a retired id fails with the current list instead of an opaque SDK error.
- **Data handling:** audio never leaves the machine; enhancement runs locally.

## References
- https://docs.ai-coustics.com
- https://artifacts.ai-coustics.io/?lang=python
