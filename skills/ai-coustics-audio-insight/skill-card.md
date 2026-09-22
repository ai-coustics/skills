# Skill card: ai-coustics-audio-insight

**Description:** Score audio with the newest ai-coustics Tyto model the installed SDK can load (family `tyto-l-16khz`) via the AIC SDK: a headline `risk_score` plus six audio-quality dimensions per 5-second window.

**Owner:** ai-coustics · **License:** MIT (skill) · the AIC SDK and model artifacts are governed by the [ai-coustics terms](https://ai-coustics.com/terms).

## Use case
Developers operating Voice AI stacks who want to rank call archives for QA review, track an audio-quality KPI, predict downstream VAD/STT/S2S failures from client-side audio, or give an LLM judge audio-quality context so agent failures can be separated from bad-audio failures in transcript-based evaluation.

## Requirements / dependencies
- **Requires credential:** yes, an AIC SDK license key via `AIC_SDK_LICENSE_KEY`, `AIC_LICENSE_KEY` or `AIC_SDK_LICENSE`. Self-service keys at https://developers.ai-coustics.com.
- **Network:** reads the model catalog and downloads the model artifact on first run to `~/.cache/aic-models`. Offline use is possible with `--model-path`.
- **Runtime:** `uv`, Python ≥ 3.11, `aic-sdk>=3.1.0`, CPU only.

Do not paste the license key into prompts, logs or output. Use least-privilege keys and rotate as appropriate.

## Skill output
- **Types:** terminal tables (stdout), optional dashboard JSON, optional per-file CSV, optional Markdown judge-context blocks.
- **Side effects:** none unless `--out-json` / `--out-csv` are passed; the model cache directory is created on first run.

## Known risks and mitigations
- **Misinterpretation:** `risk_score` measures machine impact, not perceived quality, tone or sentiment. The skill instructs the agent to say so, and the judge-context block repeats it so a downstream LLM does not misuse it.
- **Short clips:** clips under 5 s produce no score; the script exits with an explanation.
- **Data handling:** audio never leaves the machine; scoring runs locally.

## References
- https://docs.ai-coustics.com/models/audio-insight/tyto
- https://docs.ai-coustics.com/models/audio-insight/batch-call-analysis
- https://ai-coustics.com/blog/tyto-1.1
