"""
Shared helpers for ai-coustics skills: model discovery (resolve a friendly model spec to
the newest artifact id the installed SDK can load), license lookup and mono audio I/O.

This file is shipped as an identical copy in every skill's scripts/ directory so each
skill stays self-contained after `npx skills add`. The repo validator checks the
copies are byte-identical; edit one, then copy to the others.

Design goals
- Nothing here pins an SDK or model generation. The compatible artifact version is
  read from the installed SDK, and the model catalog from the public manifest at
  https://artifacts.ai-coustics.io/manifest.json at run time.
- A "family spec" is a model id with the version left out, e.g. `quail-vf-l-16khz`.
  It resolves to the newest version of that family/variant/size/rate that the SDK can
  load. Leaving out the size picks the largest at the newest version.
- An exact id (`quail-vf-2.2-l-16khz`) resolves to itself when the SDK can load it,
  and raises with the available alternatives when it cannot.
- Model type is inferred from the id prefix (`vad-*` VAD, `tyto-*` analysis, `bypass`,
  everything else enhancement). The SDK enforces the real type at load time.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

MANIFEST_URL = "https://artifacts.ai-coustics.io/manifest.json"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/aic-models")
# Formats libsndfile can decode; AAC/M4A is not among them.
SUPPORTED_EXTS = frozenset({".wav", ".flac", ".mp3", ".ogg"})
LICENSE_ENV_VARS = ("AIC_SDK_LICENSE_KEY", "AIC_LICENSE_KEY", "AIC_SDK_LICENSE")
SIZES = ("xxs", "xs", "s", "m", "l", "xl")
_VERSION_RE = re.compile(r"^\d+(?:\.\d+)+$")
_RATE_RE = re.compile(r"^(\d+)khz$")


class ModelResolutionError(Exception):
    """A spec did not match any model the installed SDK can load."""


@dataclass(frozen=True)
class ModelId:
    id: str
    family: str
    variant: str | None
    version: tuple[int, ...] | None
    size: str | None
    rate_khz: int | None

    @property
    def kind(self) -> str:
        if self.family == "vad":
            return "vad"
        if self.family == "tyto":
            return "analysis"
        if self.family == "bypass":
            return "bypass"
        return "enhancement"

    @property
    def size_rank(self) -> int:
        return SIZES.index(self.size) if self.size in SIZES else -1


def parse_model_id(model_id: str) -> ModelId | None:
    """`quail-vf-2.2-l-16khz` -> family quail, variant vf, version (2,2), size l, 16 kHz.

    Also accepts the hyphenated version spelling (`quail-vf-2-2-l-16khz`) that the
    CDN uses in file paths. Returns None for ids that do not fit the scheme.
    """
    spec = model_id.strip().lower()
    if not spec or "/" in spec or spec.endswith(".aicmodel"):
        return None
    tokens = spec.split("-")
    rate = None
    if tokens and (m := _RATE_RE.match(tokens[-1])):
        rate = int(m.group(1))
        tokens.pop()
    size = None
    if tokens and tokens[-1] in SIZES:
        size = tokens.pop()
    version: tuple[int, ...] | None = None
    # Dotted version token anywhere after the family.
    for i, tok in enumerate(tokens[1:], start=1):
        if _VERSION_RE.match(tok):
            version = tuple(int(x) for x in tok.split("."))
            del tokens[i]
            break
    else:
        # Hyphenated version: a run of >=2 purely numeric tokens after the family.
        nums = [i for i, tok in enumerate(tokens) if i > 0 and tok.isdigit()]
        if len(nums) >= 2 and nums == list(range(nums[0], nums[0] + len(nums))):
            version = tuple(int(tokens[i]) for i in nums)
            del tokens[nums[0]:nums[-1] + 1]
    if not tokens or not tokens[0]:
        return None
    family = tokens[0]
    variant = "-".join(tokens[1:]) or None
    canonical = "-".join(
        [family]
        + ([variant] if variant else [])
        + ([".".join(map(str, version))] if version else [])
        + ([size] if size else [])
        + ([f"{rate}khz"] if rate else [])
    )
    return ModelId(canonical, family, variant, version, size, rate)


def compatible_version() -> str:
    """Artifact version the installed SDK loads, e.g. 'v7'. Read, never hardcoded."""
    import aic_sdk

    return f"v{aic_sdk.get_compatible_model_version()}"


def fetch_manifest(timeout: float = 10.0) -> dict | None:
    """Return the parsed manifest, or None if it cannot be fetched or parsed.

    Retries with certifi's CA bundle when the default trust store rejects the CDN
    certificate, which happens with some Python builds.
    """
    contexts: list[ssl.SSLContext | None] = [None]
    try:
        import certifi

        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    for ctx in contexts:
        try:
            with urllib.request.urlopen(MANIFEST_URL, timeout=timeout, context=ctx) as resp:
                data = json.load(resp)
            if isinstance(data, dict) and isinstance(data.get("models"), dict):
                return data
            return None
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            # Only a certificate problem is worth a second attempt with another CA bundle;
            # a timeout or an unreachable host would just stall twice.
            if not isinstance(getattr(e, "reason", e), ssl.SSLError):
                return None
    return None


def available_models(manifest: dict, version: str) -> list[ModelId]:
    """Models in the manifest published at `version`, parsed; unparseable ids skipped."""
    out: list[ModelId] = []
    for model_id, entry in manifest.get("models", {}).items():
        versions = entry.get("versions", {}) if isinstance(entry, dict) else {}
        if version not in versions:
            continue
        parsed = parse_model_id(model_id)
        if parsed is not None:
            out.append(ModelId(model_id, parsed.family, parsed.variant, parsed.version, parsed.size, parsed.rate_khz))
    return out


def _parse_spec(spec: str, kind: str | None) -> ModelId:
    """Parse a user-supplied spec and check it is the kind of model this skill drives."""
    wanted = parse_model_id(spec)
    if wanted is None:
        raise ModelResolutionError(f"'{spec}' is not a recognisable model id or family spec")
    if kind and wanted.kind != kind:
        raise ModelResolutionError(
            f"'{spec}' is {'an' if wanted.kind[0] in 'aeiou' else 'a'} {wanted.kind} model, "
            f"but this skill drives {kind} models. "
            + {
                "vad": "Use the ai-coustics-voice-activity-detection skill for it.",
                "analysis": "Use the ai-coustics-audio-insight skill for it.",
                "enhancement": "Use the ai-coustics-speech-enhancement skill for it.",
                "bypass": "",
            }[wanted.kind]
        )
    return wanted


def _fits(m: ModelId, wanted: ModelId) -> bool:
    """True when `m` is a concrete model that the (possibly partial) spec `wanted` describes."""
    return (
        m.family == wanted.family
        and m.variant == wanted.variant
        and (wanted.size is None or m.size == wanted.size)
        and (wanted.rate_khz is None or m.rate_khz == wanted.rate_khz)
        and (wanted.version is None or m.version == wanted.version)
    )


def resolve_model(spec: str, manifest: dict, version: str, kind: str | None = None) -> str:
    """Resolve an exact id or a family spec to an artifact id the SDK can load.

    Ranking for family specs: newest version first, then largest size.
    `kind` restricts candidates to 'enhancement', 'vad', 'analysis' or 'bypass'.
    """
    wanted = _parse_spec(spec, kind)
    pool = available_models(manifest, version)
    if kind:
        pool = [m for m in pool if m.kind == kind]
    exact = [m for m in pool if m.id == wanted.id]
    if exact:
        return exact[0].id
    candidates = [m for m in pool if _fits(m, wanted)]
    if not candidates:
        alternatives = sorted(m.id for m in pool)
        raise ModelResolutionError(
            f"No model matching '{spec}' is published for the installed SDK ({version}). "
            f"Available{f' {kind}' if kind else ''} models: {', '.join(alternatives) or 'none'}"
        )
    candidates.sort(key=lambda m: (m.version or (), m.size_rank), reverse=True)
    return candidates[0].id


def resolve_or_fallback(spec: str, fallback: str, kind: str | None = None) -> str:
    """Resolve against the live manifest. If the manifest is unreachable, use an exact
    id as given, or `fallback` (the last id known to work) when it is a build of the
    family `spec` asks for; say so on stderr and let the SDK's own download step report
    anything that is wrong with it. A family spec the fallback does not satisfy raises,
    because silently running a different model would be worse than stopping."""
    manifest = fetch_manifest()
    if manifest is None:
        wanted = _parse_spec(spec, kind)
        fallback_id = parse_model_id(fallback)
        if wanted.version:
            chosen = spec
        elif fallback_id is not None and _fits(fallback_id, wanted):
            chosen = fallback
        else:
            raise ModelResolutionError(
                f"Could not read {MANIFEST_URL}, and the offline fallback '{fallback}' is not a "
                f"'{spec}' build. Pass an exact artifact id, use --model-path, or restore network access."
            )
        print(f"note: could not read {MANIFEST_URL}; using '{chosen}' without checking the catalog",
              file=sys.stderr)
        return chosen
    return resolve_model(spec, manifest, compatible_version(), kind)


def format_catalog(manifest: dict, version: str, kind: str | None = None,
                   hide: set[str] | frozenset[str] = frozenset()) -> str:
    """Human-readable list of loadable models, grouped by kind, newest first.

    `hide` drops ids from the listing (used for pre-rename ids that alias a current one).
    """
    models = [m for m in available_models(manifest, version) if m.id not in hide]
    if kind:
        models = [m for m in models if m.kind == kind]
    lines = [f"Models loadable by the installed SDK (artifact {version}):"]
    for k in ("enhancement", "vad", "analysis", "bypass"):
        group = sorted(
            (m for m in models if m.kind == k),
            key=lambda m: (m.family, m.variant or "", -(m.version[0] if m.version else 0),
                           -(m.version[1] if m.version and len(m.version) > 1 else 0), -m.size_rank),
        )
        if not group:
            continue
        lines.append(f"  {k}:")
        lines.extend(f"    {m.id}" for m in group)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runtime helpers shared by every skill script
# ---------------------------------------------------------------------------

def resolve_license(cli_key: str | None) -> str:
    """`--license-key` wins, then the env vars in LICENSE_ENV_VARS order. Never prints the key."""
    if cli_key:
        return cli_key
    for var in LICENSE_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    sys.exit(
        "No AIC SDK license key. Set one of "
        f"{', '.join(LICENSE_ENV_VARS)} or pass --license-key. "
        "Generate a self-service key at https://developers.ai-coustics.com"
    )


def download_model(artifact_id: str) -> str:
    """Download (or reuse the cached copy of) `artifact_id` into MODEL_CACHE_DIR; return its path.

    The SDK re-checks the manifest and the checksum on every call, so a retired id fails
    here with the SDK's reason rather than later with an opaque load error.
    """
    import aic_sdk as aic

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    try:
        return aic.Model.download(artifact_id, MODEL_CACHE_DIR)
    except aic.ModelDownloadError as e:
        sys.exit(f"Could not download '{artifact_id}': {e}\nRun with --list-models to see what is available.")


def load_mono_audio(path: str | Path):
    """Read any format in SUPPORTED_EXTS and mix down to a contiguous mono float32 array.

    Returns (samples, sample_rate). The SDK processes mono, so channels are averaged.
    """
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate
