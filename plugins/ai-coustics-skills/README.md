# ai-coustics-skills plugin

Claude Code / Codex / Cursor plugin wrapper around the skills in the repo's `skills/`
directory. The entries under `skills/` here are relative symlinks to the canonical
skill directories, so there is a single source of truth for every SKILL.md.

Install into Claude Code:

```
/plugin marketplace add ai-coustics/skills
/plugin install ai-coustics-skills@ai-coustics
```

Codex's local plugin installer drops symlinks. If Codex plugin distribution becomes a
goal, switch the entries here to real copies (see the `copy` mode in NVIDIA's
`plugins.d/_defaults.yml` for the pattern) or install with `npx skills add` instead.
