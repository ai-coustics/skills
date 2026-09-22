# Skill card: ai-coustics-voice-activity-detection

**Description:** Detect speech in audio files with the newest ai-coustics VAD models the installed SDK can load (VAD Multi Speaker `vad-ms-*`, VAD Voice Focus `vad-vf-*`) via the AIC SDK: per-block probabilities, decisions and speech segments.

**Owner:** ai-coustics · **License:** MIT (skill) · the AIC SDK and model artifacts are governed by the [ai-coustics terms](https://ai-coustics.com/terms).

## Use case
Developers building or debugging turn-taking and endpointing in Voice AI pipelines, segmenting recordings before transcription, or measuring speech coverage across a dataset.

## Requirements / dependencies
- **Requires credential:** yes, an AIC SDK license key via `AIC_SDK_LICENSE_KEY`, `AIC_LICENSE_KEY` or `AIC_SDK_LICENSE`. Self-service keys at https://developers.ai-coustics.com.
- **Network:** reads the model catalog and downloads the model artifact on first run to `~/.cache/aic-models`. Offline use is possible with `--model-path`.
- **Runtime:** `uv`, Python ≥ 3.11, `aic-sdk>=3.1.0`, CPU only.

Do not paste the license key into prompts, logs or output. Use least-privilege keys and rotate as appropriate.

## Skill output
- **Types:** terminal segment table and summary (stdout), optional JSON with per-block probabilities, optional segments CSV.
- **Side effects:** none unless `--out-json` / `--out-csv` are passed; the model cache directory is created on first run. Input audio is never modified.

## Known risks and mitigations
- **Threshold direction:** sensitivity is a probability threshold where higher detects less, the opposite of the retired energy VAD. The skill states this explicitly.
- **Micro-segments:** with the model default of 0 s minimum speech duration, brief clicks can register. The skill documents `--min-speech`.
- **Data handling:** audio never leaves the machine; detection runs locally.

## References
- https://docs.ai-coustics.com/models/voice-activity-detection/vad
- https://docs.ai-coustics.com/reference/sdk/models
- https://docs.ai-coustics.com/reference/deprecated/energy-vad-to-dedicated-vad
