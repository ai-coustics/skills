# ai-coustics Agent Skills

Skills that teach AI coding agents (Claude Code, Codex, Cursor and others) how to use
ai-coustics speech-enhancement and audio-insight models through the AIC SDK.

Skills follow the [Agent Skills](https://agentskills.io/) format: each one is a
directory with a `SKILL.md` that the agent loads on demand, plus the scripts and
references it needs.

## Install

```bash
npx skills add ai-coustics/skills                                  # interactive picker
npx skills add ai-coustics/skills --skill ai-coustics-speech-enhancement   # one skill
npx skills add ai-coustics/skills --skill ai-coustics-audio-insight --agent claude-code
```

Claude Code users can also install the bundle as a plugin:

```
/plugin marketplace add ai-coustics/skills
/plugin install ai-coustics-skills@ai-coustics
```

Manual install: copy `skills/<name>` into `~/.claude/skills/` (Claude Code) or the
equivalent skills directory of your agent.

## Prerequisites

Every skill here runs the AIC SDK locally. You need:

- `uv` on PATH. Each script declares its dependencies inline and runs with `uv run`.
- An AIC SDK license key exported as `AIC_SDK_LICENSE_KEY` (the scripts also accept
  `AIC_LICENSE_KEY` and `AIC_SDK_LICENSE`). Generate a self-service key at
  https://developers.ai-coustics.com.

Models download on first use to `~/.cache/aic-models`. Audio never leaves your machine.

## Skills

One skill per [ai-coustics.com](https://ai-coustics.com/) product line.

### ai-coustics-speech-enhancement

Enhance speech recordings, single file or whole folder, with Quail Voice Focus, Quail
Multi Speaker or Rook Multi Speaker. Async worker pool, latency-compensated output of
identical length, mono at the input sample rate.

**Use when:**

- "Enhance this recording with the AIC SDK"
- "Enhance every call under ./raw/"
- "Apply voice focus at level 0.7"

### ai-coustics-audio-insight

Score audio with the Tyto model: a headline `risk_score` plus per-window quality
dimensions such as noise, reverb, interfering speech, codec degradation and packet loss,
read from whatever the installed SDK exposes. Predicts whether client-side audio will break VAD,
turn-taking, STT or speech-to-speech downstream.

**Use when:**

- "Score this call with Tyto" / "What's the risk score?"
- "Rank this folder of calls for QA review"
- "Generate the JSON for the call-analysis dashboard"
- "Our LLM judge blames the agent; was it the audio?" (writes judge-prompt context with degraded intervals)
- Explaining what a Tyto dimension means

### ai-coustics-voice-activity-detection

Detect speech with the VAD Multi Speaker or VAD Voice Focus models: per-block
probabilities, post-processed decisions, and speech segments with delay-compensated
timestamps, as a table, CSV or JSON.

**Use when:**

- "Find the speech segments in this recording"
- "How much of these files is actually speech?"
- "Tune the VAD so our agent stops interrupting callers"
- "Detect only the main speaker, ignore the TV in the background"

## Repository structure

```
skills/                   one directory per skill, kebab-case, ai-coustics- prefixed
  <name>/
    SKILL.md              frontmatter (name, description) + agent instructions
    scripts/              uv-runnable scripts with inline dependencies
      _aic_models.py      shared model-discovery helper (identical copy per skill)
    references/           deep material loaded on demand (optional)
    evals/evals.json      trigger + behaviour test cases
    skill-card.md         owner, credentials, output, risks
plugins/ai-coustics-skills/       Claude Code plugin bundling the catalog (symlinks into skills/)
.claude-plugin/           marketplace manifest
skills.sh.json            grouping for the skills.sh directory
scripts/validate_skills.py  frontmatter, naming, pin and helper-sync checks (runs in CI)
tests/                    pytest for the scripts, heavy deps stubbed in conftest.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checklist and [AGENTS.md](AGENTS.md)
for the authoring conventions agents follow when adding a skill here.

## License

MIT for the skills and scripts in this repository. Use of the AIC SDK and model
artifacts is governed by the ai-coustics terms.
