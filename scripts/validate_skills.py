# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6"]
# ///
"""
Validate the skills catalog.

Checks every skills/<dir>/SKILL.md for spec-conformant frontmatter, that the
directory name matches `name`, that scripts and references mentioned in SKILL.md
exist, that evals/evals.json is well-formed, that inline script dependencies are
floors (>=) rather than pins (==), that every copy of the shared _aic_models.py
helper is identical, and that skills.sh.json and the plugin symlinks only point at
skills that exist. Exit code 1 on any failure.

Usage:
  uv run scripts/validate_skills.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = ROOT / "skills"
PLUGIN_SKILLS_DIR = ROOT / "plugins" / "ai-coustics-skills" / "skills"
SKILLS_SH = ROOT / "skills.sh.json"

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NAME_PREFIX = "ai-coustics-"
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_SKILL_LINES = 300
SHARED_HELPER = "_aic_models.py"  # identical copy in every skill's scripts/
MIN_POSITIVE_EVALS = 3  # AGENTS.md: three or more positive cases ...
MIN_NEGATIVE_EVALS = 1  # ... and at least one negative case
REQUIRED_EVAL_KEYS = {"id", "question", "expected_skill", "expected_script", "ground_truth", "expected_behavior"}
# {BASE_DIR}/scripts/foo.py  or  references/bar.md  or  scripts/foo.py
LOCAL_REF_RE = re.compile(r"(?:\{BASE_DIR\}/)?((?:scripts|references)/[A-Za-z0-9_./-]+\.(?:py|md|sh|json))")

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def parse_frontmatter(text: str, where: str) -> dict | None:
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if not m:
        err(f"{where}: missing YAML frontmatter")
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        err(f"{where}: invalid frontmatter YAML: {e}")
        return None
    if not isinstance(data, dict):
        err(f"{where}: frontmatter is not a mapping")
        return None
    return data


def check_skill(skill_dir: Path) -> str | None:
    where = skill_dir.relative_to(ROOT)
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        err(f"{where}: SKILL.md missing")
        return None

    text = skill_md.read_text(encoding="utf-8")
    fm = parse_frontmatter(text, f"{where}/SKILL.md")
    if fm is None:
        return None

    name = fm.get("name")
    desc = fm.get("description")
    if not isinstance(name, str) or not name:
        err(f"{where}/SKILL.md: `name` missing")
        name = None
    else:
        if name != skill_dir.name:
            err(f"{where}/SKILL.md: name `{name}` != directory `{skill_dir.name}`")
        if not NAME_RE.match(name) or len(name) > MAX_NAME:
            err(f"{where}/SKILL.md: name `{name}` is not kebab-case or exceeds {MAX_NAME} chars")
        if not name.startswith(NAME_PREFIX):
            err(f"{where}/SKILL.md: name `{name}` must start with `{NAME_PREFIX}`")
    if not isinstance(desc, str) or not desc.strip():
        err(f"{where}/SKILL.md: `description` missing")
    elif len(desc) > MAX_DESCRIPTION:
        err(f"{where}/SKILL.md: description is {len(desc)} chars (max {MAX_DESCRIPTION})")
    if "license" not in fm:
        err(f"{where}/SKILL.md: `license` missing from frontmatter")
    meta = fm.get("metadata")
    if not isinstance(meta, dict) or "version" not in meta:
        err(f"{where}/SKILL.md: `metadata.version` missing")

    n_lines = text.count("\n") + 1
    if n_lines > MAX_SKILL_LINES:
        err(f"{where}/SKILL.md: {n_lines} lines (max {MAX_SKILL_LINES}); move depth into references/")

    for ref in sorted(set(LOCAL_REF_RE.findall(text))):
        if not (skill_dir / ref).exists():
            err(f"{where}/SKILL.md: references `{ref}` which does not exist")

    evals = skill_dir / "evals" / "evals.json"
    if not evals.is_file():
        err(f"{where}: evals/evals.json missing")
    else:
        try:
            cases = json.loads(evals.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err(f"{where}/evals/evals.json: invalid JSON: {e}")
            cases = []
        if not isinstance(cases, list) or not cases:
            err(f"{where}/evals/evals.json: must be a non-empty list")
            cases = []
        ids: set[str] = set()
        positives = negatives = 0
        for i, case in enumerate(cases):
            if not isinstance(case, dict):
                err(f"{where}/evals/evals.json[{i}]: not an object")
                continue
            missing = REQUIRED_EVAL_KEYS - case.keys()
            if missing:
                err(f"{where}/evals/evals.json[{i}]: missing keys {sorted(missing)}")
            cid = case.get("id")
            if cid in ids:
                err(f"{where}/evals/evals.json: duplicate id `{cid}`")
            ids.add(cid)
            if case.get("expected_skill") == name:
                positives += 1
            elif "expected_skill" in case:
                negatives += 1
            script = case.get("expected_script")
            if script and not (skill_dir / script).exists():
                err(f"{where}/evals/evals.json[{i}]: expected_script `{script}` does not exist")
        if name and cases:
            if positives < MIN_POSITIVE_EVALS:
                err(f"{where}/evals/evals.json: {positives} case(s) with expected_skill == `{name}` "
                    f"(need at least {MIN_POSITIVE_EVALS})")
            if negatives < MIN_NEGATIVE_EVALS:
                err(f"{where}/evals/evals.json: {negatives} negative case(s) (expected_skill != `{name}`; "
                    f"need at least {MIN_NEGATIVE_EVALS})")

    card = skill_dir / "skill-card.md"
    if not card.is_file():
        err(f"{where}: skill-card.md missing (every AIC SDK skill needs a credential card)")

    # Version agnosticism: inline script dependencies may set floors (>=) but never pins (==).
    for script in sorted((skill_dir / "scripts").glob("*.py")) if (skill_dir / "scripts").is_dir() else []:
        text_s = script.read_text(encoding="utf-8")
        block = re.search(r"^# /// script\n(.*?)^# ///", text_s, re.S | re.M)
        if block:
            for line in block.group(1).splitlines():
                if "==" in line and not line.strip().startswith("# requires-python"):
                    err(f"{script.relative_to(ROOT)}: pinned dependency `{line.strip('# ').strip()}`; use a >= floor")
        if "from _aic_models import" in text_s and not (skill_dir / "scripts" / SHARED_HELPER).is_file():
            err(f"{script.relative_to(ROOT)}: imports {SHARED_HELPER} but the skill has no copy of it")

    return name


def main() -> int:
    if not SKILLS_DIR.is_dir():
        err("skills/ directory missing")
        return report()

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))
    names = {n for n in (check_skill(d) for d in skill_dirs) if n}

    # Shared helper: every copy must be byte-identical to every other.
    copies = sorted(SKILLS_DIR.glob(f"*/scripts/{SHARED_HELPER}"))
    if len(copies) > 1:
        reference = copies[0].read_bytes()
        for c in copies[1:]:
            if c.read_bytes() != reference:
                err(f"{c.relative_to(ROOT)}: differs from {copies[0].relative_to(ROOT)}; "
                    f"edit one copy and cp it to the others")

    # skills.sh.json groupings must reference real skills, each at most once.
    if SKILLS_SH.is_file():
        try:
            cfg = json.loads(SKILLS_SH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            err(f"skills.sh.json: invalid JSON: {e}")
            cfg = {}
        seen: set[str] = set()
        for group in cfg.get("groupings", []):
            for s in group.get("skills", []):
                if s not in names:
                    err(f"skills.sh.json: group `{group.get('title')}` lists unknown skill `{s}`")
                if s in seen:
                    err(f"skills.sh.json: skill `{s}` appears in more than one group")
                seen.add(s)
        for n in sorted(names - seen):
            err(f"skills.sh.json: skill `{n}` is not in any group")
    else:
        err("skills.sh.json missing")

    # Plugin bundle: each entry resolves to a catalog skill, and every skill is bundled.
    if PLUGIN_SKILLS_DIR.is_dir():
        bundled: set[str] = set()
        for entry in sorted(PLUGIN_SKILLS_DIR.iterdir()):
            target = entry.resolve()
            if not (target / "SKILL.md").is_file():
                err(f"plugins/ai-coustics-skills/skills/{entry.name}: does not resolve to a skill directory")
                continue
            if target.parent != SKILLS_DIR.resolve():
                err(f"plugins/ai-coustics-skills/skills/{entry.name}: points outside skills/ ({target})")
            if entry.name != target.name:
                err(f"plugins/ai-coustics-skills/skills/{entry.name}: link name differs from target `{target.name}`")
            bundled.add(target.name)
        for n in sorted(names - bundled):
            err(f"plugins/ai-coustics-skills/skills: skill `{n}` is not bundled (add a symlink)")
    else:
        err("plugins/ai-coustics-skills/skills/ missing")

    return report(len(names))


def report(n_skills: int = 0) -> int:
    if errors:
        print(f"FAIL: {len(errors)} problem(s)", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"OK: {n_skills} skill(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
