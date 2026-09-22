# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, Copilot) working in this
repository.

## Repository overview

A public catalog of Agent Skills for ai-coustics products. Every skill wraps the AIC
SDK: it needs a license key, downloads a model on first run, and processes audio
locally. Skills are loaded on demand: only `name` and `description` are visible at
startup, the rest of `SKILL.md` loads when the agent picks the skill.

## Creating a new skill

### Directory layout

```
skills/
  ai-coustics-{skill-name}/       # kebab-case, always ai-coustics- prefixed
    SKILL.md              # required
    scripts/              # uv-runnable .py scripts with PEP 723 inline deps
      _aic_models.py      # copy of the shared model-discovery helper
    references/           # optional, loaded on demand from SKILL.md
    evals/evals.json      # required: trigger + behaviour cases
    skill-card.md         # required when the skill needs a credential
tests/
  test_ai_coustics_{skill_name}.py   # pytest, imports the script by path
```

Then register the skill in `skills.sh.json` and add a symlink under
`plugins/ai-coustics-skills/skills/` pointing at `../../../skills/ai-coustics-{skill-name}`.

### SKILL.md frontmatter

```yaml
---
name: ai-coustics-{skill-name}          # must equal the directory name
description: >                  # <= 1024 chars, written as trigger phrases
  What it does in one sentence. Trigger this skill when the user asks to: ...
  Also trigger when the user references "...", "...".
license: MIT
metadata:
  author: ai-coustics
  version: "1.0.0"
---
```

### SKILL.md body

Sections, in order: title and one-paragraph summary, Prerequisites, Steps, domain
reference (models, metrics), script arguments table, Output, Key implementation
details, Troubleshooting, Reference links. Keep the file under 300 lines (the validator enforces this) and push
depth into `references/`.

Write `{BASE_DIR}` for the skill's own directory when showing commands. Claude Code
prints it when the skill loads; other agents resolve it to the install location.

### Version agnosticism (non-negotiable)

Skills must work with the newest SDK and the newest models without edits.

- Inline dependencies are floors (`aic-sdk>=3.1.0`), never pins. The validator fails on `==`.
- Never hardcode a concrete model id as the default. Use a *family spec* (id without the
  version, e.g. `quail-vf-l-16khz`) and resolve it through `scripts/_aic_models.py`,
  which reads `aic.get_compatible_model_version()` and the public manifest at run time.
  Keep one `FALLBACK_MODEL` constant for when the manifest is unreachable, clearly
  labelled as last-known-good.
- Read block sizes, delays, parameter defaults and result fields from the SDK or the
  model, never from constants. Tyto's dimensions come from `dir(aic.AnalysisResult)`.
- Offer `--list-models` so the agent can answer "which models exist" from live data.
- SKILL.md must not state model versions, artifact versions ("v7") or retired-model lists
  as facts. Describe families, aliases and how to look up what is current.
- `_aic_models.py` is copied byte-for-byte into every skill so each stays self-contained
  after install. Edit one copy, `cp` it to the others; the validator checks they match.

### Script conventions

- Start with a PEP 723 block so `uv run script.py` needs no environment setup. Declare
  `aic-sdk>=3.1.0` and `certifi` (the helper needs it for the manifest fetch).
- Accept the license key from `--license-key` or the env vars
  `AIC_SDK_LICENSE_KEY`, `AIC_LICENSE_KEY`, `AIC_SDK_LICENSE`. Exit with a clear
  message pointing at https://developers.ai-coustics.com when none is set. Never
  print the key.
- Cache models in `~/.cache/aic-models`. Offer `--model-path` for offline use.
- Accept both a single file and a folder. Folder mode mirrors the input tree.
- Human-readable progress and tables go to stdout. Machine-readable results are
  opt-in via `--out-json` / `--out-csv`; never write files the user did not ask for.
- Fail fast with an explanation for known-bad inputs (retired models, VAD artifacts,
  clips too short to score) instead of surfacing an SDK traceback.

### Tests and evals

- `tests/test_ai_coustics_{name}.py` loads the script with `importlib` and tests pure
  functions (alias resolution, aggregation, writers). `conftest.py` stubs `aic_sdk`,
  `soundfile`, `tqdm` and other heavy imports so CI needs no license key or model.
- `evals/evals.json` holds three or more positive cases and at least one negative
  case (a prompt that should trigger a different skill or none). Each case has
  `id`, `question`, `expected_skill`, `expected_script`, `ground_truth`,
  `expected_behavior`.

### Before opening a PR

```bash
uv run scripts/validate_skills.py
uv run --with "pytest>=8" --with "numpy>=1.24" --with "pyyaml>=6" pytest tests/ -v
```

## Terminology

- The models do **speech enhancement**. Never write "denoise", "denoising" or "noise
  suppression" for what they do; say "enhance speech", "speech enhancement", "voice
  focus" or "isolate the primary speaker". Naming the `noise` dimension of Tyto is fine.
- Audio going from the caller into the agent is **client-side** audio, not "contact-side".
- `ms` is Multi Speaker, `vf` is Voice Focus.

## Never

- Do not paste secrets, license keys, or customer audio paths into SKILL.md or evals.
- Do not reference internal ai-coustics tooling, repos, or datasets. If a skill needs
  a helper from an internal package, inline or vendor the minimal code instead.
- Do not add agent-specific directories (`.claude/skills/`, `.cursor/skills/`). The
  `skills/` tree is the source of truth; installers copy from it.
