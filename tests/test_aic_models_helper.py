"""Tests for the shared _aic_models.py helper (one copy per skill; validator keeps them identical)."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
HELPER = REPO / "skills/ai-coustics-speech-enhancement/scripts/_aic_models.py"

# Loaded under a private name so this does not replace the `_aic_models` module the
# skill scripts import (their tests monkeypatch that one via sys.modules["_aic_models"]).
_spec = importlib.util.spec_from_file_location("_aic_models_under_test", HELPER)
am = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = am
_spec.loader.exec_module(am)

# A manifest shaped like the real one, with a mix of generations so ranking is exercised.
MANIFEST = {
    "schema_version": "1.0",
    "models": {
        "quail-vf-2.2-l-16khz": {"versions": {"v6": {}, "v7": {}}},
        "quail-vf-2.2-s-16khz": {"versions": {"v7": {}}},
        "quail-vf-2.3-l-16khz": {"versions": {"v8": {}}},          # next generation, not loadable at v7
        "quail-vf-2.1-l-16khz": {"versions": {"v5": {}}},          # retired
        "quail-vf-2.1-l-48khz": {"versions": {"v7": {}}},
        "quail-ms-l-16khz": {"versions": {"v7": {}}},
        "quail-l-16khz": {"versions": {"v7": {}}},
        "rook-ms-s-48khz": {"versions": {"v7": {}}},
        "vad-ms-2.1-xxs-16khz": {"versions": {"v7": {}}},
        "vad-2.1-xxs-16khz": {"versions": {"v7": {}}},
        "vad-vf-2.0-s-16khz": {"versions": {"v7": {}}},
        "tyto-1.1-l-16khz": {"versions": {"v7": {}}},
        "tyto-1.2-l-16khz": {"versions": {"v7": {}}},
        "bypass": {"versions": {"v7": {}}},
        "weird/id.aicmodel": {"versions": {"v7": {}}},
    },
}


def test_parse_model_id_full():
    m = am.parse_model_id("quail-vf-2.2-l-16khz")
    assert (m.family, m.variant, m.version, m.size, m.rate_khz) == ("quail", "vf", (2, 2), "l", 16)
    assert m.kind == "enhancement" and m.id == "quail-vf-2.2-l-16khz"


def test_parse_model_id_hyphenated_version_canonicalises():
    assert am.parse_model_id("quail-vf-2-2-l-16khz").id == "quail-vf-2.2-l-16khz"
    assert am.parse_model_id("vad-ms-2-1-xxs-16khz").id == "vad-ms-2.1-xxs-16khz"


def test_parse_model_id_variants_and_kinds():
    assert am.parse_model_id("vad-ms-2.1-xxs-16khz").kind == "vad"
    assert am.parse_model_id("vad-2.1-xxs-16khz").variant is None
    assert am.parse_model_id("tyto-1.1-l-16khz").kind == "analysis"
    assert am.parse_model_id("bypass").kind == "bypass"
    assert am.parse_model_id("rook-ms-s-48khz").version is None
    fam = am.parse_model_id("quail-vf-l-16khz")
    assert fam.version is None and fam.size == "l"
    assert am.parse_model_id("quail-vf-16khz").size is None
    assert am.parse_model_id("weird/id.aicmodel") is None
    assert am.parse_model_id("") is None


def test_available_models_filters_by_version():
    ids = {m.id for m in am.available_models(MANIFEST, "v7")}
    assert "quail-vf-2.2-l-16khz" in ids
    assert "quail-vf-2.3-l-16khz" not in ids  # v8 only
    assert "quail-vf-2.1-l-16khz" not in ids  # v5 only
    assert "weird/id.aicmodel" not in ids


def test_resolve_family_spec_picks_newest_then_largest():
    assert am.resolve_model("quail-vf-l-16khz", MANIFEST, "v7") == "quail-vf-2.2-l-16khz"
    assert am.resolve_model("quail-vf-16khz", MANIFEST, "v7") == "quail-vf-2.2-l-16khz"   # size omitted -> l
    assert am.resolve_model("quail-vf-s-16khz", MANIFEST, "v7") == "quail-vf-2.2-s-16khz"
    assert am.resolve_model("quail-vf-l-48khz", MANIFEST, "v7") == "quail-vf-2.1-l-48khz"
    assert am.resolve_model("tyto-l-16khz", MANIFEST, "v7") == "tyto-1.2-l-16khz"
    assert am.resolve_model("vad-ms-16khz", MANIFEST, "v7") == "vad-ms-2.1-xxs-16khz"
    assert am.resolve_model("vad-vf-16khz", MANIFEST, "v7") == "vad-vf-2.0-s-16khz"
    # A newer SDK generation sees the newer model with no code change.
    assert am.resolve_model("quail-vf-l-16khz", MANIFEST, "v8") == "quail-vf-2.3-l-16khz"


def test_resolve_exact_id_and_hyphen_spelling():
    assert am.resolve_model("quail-vf-2.2-l-16khz", MANIFEST, "v7") == "quail-vf-2.2-l-16khz"
    assert am.resolve_model("quail-vf-2-2-l-16khz", MANIFEST, "v7") == "quail-vf-2.2-l-16khz"
    assert am.resolve_model("bypass", MANIFEST, "v7") == "bypass"
    assert am.resolve_model("quail-l-16khz", MANIFEST, "v7") == "quail-l-16khz"  # exact wins over family


def test_resolve_kind_filter_rejects_wrong_type():
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_model("vad-ms-16khz", MANIFEST, "v7", kind="enhancement")
    assert "is a vad model" in str(e.value)
    assert am.resolve_model("vad-ms-16khz", MANIFEST, "v7", kind="vad") == "vad-ms-2.1-xxs-16khz"
    # Right kind, but a family the manifest lacks -> lists the alternatives of that kind.
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_model("owl-l-16khz", MANIFEST, "v7", kind="enhancement")
    assert "Available enhancement models" in str(e.value)


def test_resolve_retired_id_lists_alternatives():
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_model("quail-vf-2.1-l-16khz", MANIFEST, "v7")
    assert "quail-vf-2.2-l-16khz" in str(e.value)


def test_resolve_garbage():
    with pytest.raises(am.ModelResolutionError):
        am.resolve_model("", MANIFEST, "v7")
    with pytest.raises(am.ModelResolutionError):
        am.resolve_model("nonexistent-l-16khz", MANIFEST, "v7")


def test_resolve_or_fallback_offline(monkeypatch, capsys):
    monkeypatch.setattr(am, "fetch_manifest", lambda: None)
    # Family spec offline -> last-known fallback; exact id offline -> itself.
    assert am.resolve_or_fallback("quail-vf-l-16khz", "quail-vf-2.2-l-16khz") == "quail-vf-2.2-l-16khz"
    assert am.resolve_or_fallback("quail-vf-2.1-l-48khz", "x") == "quail-vf-2.1-l-48khz"
    assert "could not read" in capsys.readouterr().err


def test_resolve_or_fallback_online(monkeypatch):
    monkeypatch.setattr(am, "fetch_manifest", lambda: MANIFEST)
    monkeypatch.setattr(am, "compatible_version", lambda: "v7")
    assert am.resolve_or_fallback("tyto-l-16khz", "tyto-1.1-l-16khz", kind="analysis") == "tyto-1.2-l-16khz"


def test_format_catalog_groups_by_kind():
    text = am.format_catalog(MANIFEST, "v7")
    assert "enhancement:" in text and "vad:" in text and "analysis:" in text
    assert text.index("tyto-1.2-l-16khz") < text.index("tyto-1.1-l-16khz")  # newest first
    only_vad = am.format_catalog(MANIFEST, "v7", kind="vad")
    assert "quail" not in only_vad and "vad-vf-2.0-s-16khz" in only_vad


def test_resolve_wrong_kind_names_the_right_skill():
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_model("vad-ms-16khz", MANIFEST, "v7", kind="enhancement")
    assert "is a vad model" in str(e.value) and "voice-activity-detection" in str(e.value)
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_model("quail-vf-l-16khz", MANIFEST, "v7", kind="analysis")
    assert "speech-enhancement" in str(e.value)


def test_format_catalog_hide():
    text = am.format_catalog(MANIFEST, "v7", kind="enhancement", hide={"quail-l-16khz"})
    assert "quail-l-16khz" not in text and "quail-ms-l-16khz" in text


def test_resolve_or_fallback_offline_refuses_other_family(monkeypatch):
    monkeypatch.setattr(am, "fetch_manifest", lambda: None)
    # Fallback is a build of the requested family: fine.
    assert am.resolve_or_fallback("vad-ms-16khz", "vad-ms-2.1-xxs-16khz", kind="vad") == "vad-ms-2.1-xxs-16khz"
    # Fallback is a different variant: refuse instead of silently running Multi Speaker as Voice Focus.
    with pytest.raises(am.ModelResolutionError) as e:
        am.resolve_or_fallback("vad-vf-16khz", "vad-ms-2.1-xxs-16khz", kind="vad")
    assert "offline fallback" in str(e.value)
    # The kind check still applies offline.
    with pytest.raises(am.ModelResolutionError):
        am.resolve_or_fallback("quail-vf-2.2-l-16khz", "vad-ms-2.1-xxs-16khz", kind="vad")


def test_fetch_manifest_retries_only_on_certificate_errors(monkeypatch):
    import ssl
    import types
    import urllib.error
    # CI has no certifi; pretend it is there so a second, certifi-backed context exists.
    monkeypatch.setitem(sys.modules, "certifi", types.SimpleNamespace(where=lambda: "certifi.pem"))
    monkeypatch.setattr(am.ssl, "create_default_context", lambda cafile=None: object())
    calls = []

    def urlopen_timeout(*_a, **_k):
        calls.append("timeout")
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(am.urllib.request, "urlopen", urlopen_timeout)
    assert am.fetch_manifest() is None
    assert calls == ["timeout"]  # no pointless second attempt with another CA bundle

    calls.clear()

    def urlopen_cert(*_a, **_k):
        calls.append("cert")
        raise urllib.error.URLError(ssl.SSLCertVerificationError("bad cert"))

    monkeypatch.setattr(am.urllib.request, "urlopen", urlopen_cert)
    assert am.fetch_manifest() is None
    assert len(calls) == 2  # default trust store, then certifi

    calls.clear()

    def urlopen_incomplete(*_a, **_k):
        import http.client
        calls.append("incomplete")
        raise http.client.IncompleteRead(b"")

    monkeypatch.setattr(am.urllib.request, "urlopen", urlopen_incomplete)
    assert am.fetch_manifest() is None  # HTTPException is handled, not raised
