# Contributing

Thanks for helping make ai-coustics models easier for agents to use.

## Reporting issues

Open a GitHub issue for bugs in a skill, unclear instructions, or a model or SDK change
the skill has not caught up with. Include the agent you used and the prompt that
triggered the skill. Please do not include license keys or customer audio.

## Proposing a skill

1. Open an issue first describing the workflow the skill covers and which AIC SDK
   surface it uses. This avoids two people building the same thing.
2. Follow the layout and conventions in [AGENTS.md](AGENTS.md). In short: one
   directory under `skills/` with a `SKILL.md`, `scripts/`, `evals/evals.json` and,
   because every skill here needs a license key, a `skill-card.md`.
3. Add a pytest file under `tests/` and register the skill in `skills.sh.json` and
   `plugins/ai-coustics-skills/skills/`.
4. Run the checks locally:

   ```bash
   uv run scripts/validate_skills.py
   uv run --with "pytest>=8" --with "numpy>=1.24" --with "pyyaml>=6" pytest tests/ -v
   ```

5. Open a pull request. CI runs the same two commands.

## Changing an existing skill

Bump `metadata.version` in the skill's frontmatter (semver: patch for wording, minor
for new flags or models, major for changed defaults or removed behaviour). Update the
evals if the trigger phrases or expected behaviour changed.

## Style

- Descriptions are trigger phrases, not marketing copy.
- Prefer scripts over inline code in SKILL.md; script execution costs no context.
- Keep SKILL.md under 300 lines (validator-enforced); move depth into `references/`.
- Public only: no internal repos, datasets, or tooling.
- Version-agnostic: no pinned SDK versions, no hardcoded model versions; see AGENTS.md.

## License

By contributing you agree that your contribution is licensed under the MIT license
in [LICENSE](LICENSE).
