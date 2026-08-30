"""
FR-02 — Projection: embed text, score dimensions against a user schema.

Written test-first per GOV-04: every test here was RED before
kse_memory/core/schema.py and kse_memory/core/projection.py existed.

Acceptance criteria encoded (BD4):
- TC-04 (US4): a YAML schema of named dimensions with anchors is loaded,
  dimensions are scored and queryable, and no hardcoded fashion vocabulary
  remains in the default path.
- TC-07 (US7): with no API key, the local scorer produces schema-conformant
  scores.
- TC-02 (US2) / AR-01: the default path makes zero network calls.
- BD4 "Replay": content hash + schema version + model IDs reproduce any
  projection.

Scope note: FR-02's third limb — incremental graph-edge upsert — is not
covered here. It depends on GraphStoreInterface and is specified by TC-09,
so it is deliberately left to its own TC cycle rather than half-tested here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from kse_memory.core.ingest import content_hash, normalise_record
from kse_memory.core.projection import (
    ModelNotAvailableError,
    default_cache_dir,
    default_model_path,
    OnnxEmbedder,
    Projection,
    project,
)
from kse_memory.core.schema import DimensionSchema, SchemaError, load_schema


# The deterministic stub embedder lives in conftest (D-16, T-065): a real
# implementation of the embedder contract, shared by every offline lane.
from tests.conftest import StubEmbedder

SCHEMA_YAML = """
name: generic-v1
version: 1.0.0
dimensions:
  - name: technical_depth
    description: How technical the material is
    anchors:
      - dense technical specification with precise terminology
      - implementation detail aimed at engineers
  - name: accessibility
    description: How approachable the material is for a newcomer
    anchors:
      - plain language introduction for a general audience
      - gentle explanation assuming no prior knowledge
"""


@pytest.fixture
def schema(tmp_path):
    path = tmp_path / "schema.yaml"
    path.write_text(SCHEMA_YAML, encoding="utf-8")
    return load_schema(path)


@pytest.fixture
def entity():
    return normalise_record(
        {
            "title": "Vector index internals",
            "description": "HNSW graph construction and ef_search tuning.",
            "tags": ["ann", "index"],
        }
    )


# ------------------------------------------------------------------ US4 schema
def test_load_schema_from_yaml_file(schema):
    assert isinstance(schema, DimensionSchema)
    assert schema.name == "generic-v1"
    assert schema.version == "1.0.0"
    assert schema.names() == ("technical_depth", "accessibility")
    assert schema["technical_depth"].anchors  # anchors are retained for scoring


def test_load_schema_from_mapping():
    s = load_schema(
        {
            "name": "inline",
            "version": "0.1.0",
            "dimensions": [{"name": "d", "description": "x", "anchors": ["a"]}],
        }
    )
    assert s.names() == ("d",)


def test_schema_rejects_duplicate_dimension_names():
    with pytest.raises(SchemaError, match="duplicate"):
        load_schema(
            {
                "name": "dup",
                "version": "1.0.0",
                "dimensions": [
                    {"name": "d", "description": "x", "anchors": ["a"]},
                    {"name": "d", "description": "y", "anchors": ["b"]},
                ],
            }
        )


def test_schema_rejects_dimension_without_anchors():
    with pytest.raises(SchemaError, match="anchor"):
        load_schema(
            {
                "name": "noanchor",
                "version": "1.0.0",
                "dimensions": [{"name": "d", "description": "x", "anchors": []}],
            }
        )


def test_schema_rejects_non_semver_version():
    with pytest.raises(SchemaError, match="version"):
        load_schema(
            {
                "name": "badver",
                "version": "v1",
                "dimensions": [{"name": "d", "description": "x", "anchors": ["a"]}],
            }
        )


def test_schema_rejects_empty_dimension_set():
    with pytest.raises(SchemaError, match="dimension"):
        load_schema({"name": "empty", "version": "1.0.0", "dimensions": []})


# ------------------------------------------------------------- TC-04 / TC-07
def test_project_scores_every_schema_dimension(schema, entity):
    """TC-04: dimensions are scored and queryable by name."""
    p = project(entity, schema, StubEmbedder())
    assert isinstance(p, Projection)
    assert set(p.scores) == set(schema.names())


def test_scores_are_bounded_unit_interval(schema, entity):
    """TC-07: the local scorer produces schema-conformant scores, no API key."""
    p = project(entity, schema, StubEmbedder())
    assert all(0.0 <= v <= 1.0 for v in p.scores.values()), p.scores


def test_projection_carries_replay_identity(schema, entity):
    """BD4 Replay: content hash + schema version + model id reproduce a projection."""
    p = project(entity, schema, StubEmbedder())
    assert p.content_hash == content_hash(entity)
    assert p.schema_name == schema.name
    assert p.schema_version == schema.version
    assert p.model_id == StubEmbedder.model_id


def test_projection_is_deterministic(schema, entity):
    a = project(entity, schema, StubEmbedder())
    b = project(entity, schema, StubEmbedder())
    assert a == b


def test_projection_changes_when_content_changes(schema):
    a = project(normalise_record({"title": "t", "description": "alpha"}), schema, StubEmbedder())
    b = project(normalise_record({"title": "t", "description": "beta"}), schema, StubEmbedder())
    assert a.scores != b.scores


def test_projection_survives_tag_reorder(schema):
    """Ties to FR-01: cosmetic reorder must not force re-projection."""
    a = project(normalise_record({"title": "t", "description": "d", "tags": ["x", "y"]}), schema, StubEmbedder())
    b = project(normalise_record({"title": "t", "description": "d", "tags": ["y", "x"]}), schema, StubEmbedder())
    assert a == b


def test_schema_version_participates_in_identity(schema, entity, tmp_path):
    """A schema bump must be visible in the projection, or replay is a lie."""
    bumped_path = tmp_path / "bumped.yaml"
    bumped_path.write_text(SCHEMA_YAML.replace("version: 1.0.0", "version: 1.1.0"), encoding="utf-8")
    bumped = load_schema(bumped_path)
    a = project(entity, schema, StubEmbedder())
    b = project(entity, bumped, StubEmbedder())
    assert a.schema_version != b.schema_version
    assert a != b


# ------------------------------------------------------------------- AR-01
def test_projection_makes_no_network_calls(no_network, schema, entity):
    """AR-01: the whole default projection path is socket-free."""
    p = project(entity, schema, StubEmbedder())
    assert p.scores


def test_onnx_embedder_never_downloads(no_network, tmp_path):
    """AR-01: a missing local model fails loudly; it must never fetch one."""
    missing = tmp_path / "definitely-absent.onnx"
    with pytest.raises(ModelNotAvailableError, match="local"):
        OnnxEmbedder(model_path=missing)


def test_onnx_embedder_reports_the_path_it_wanted(tmp_path):
    """The error must name the path, or the CPU-only setup story is unusable."""
    missing = tmp_path / "absent.onnx"
    with pytest.raises(ModelNotAvailableError) as exc:
        OnnxEmbedder(model_path=missing)
    assert str(missing) in str(exc.value)


# ------------------------------------------------------------------- TC-04
_DOMAIN_VOCABULARY = (
    "elegance",
    "comfort",
    "boldness",
    "modernity",
    "minimalism",
    "luxury",
    "seasonality",
)


def test_default_path_has_no_hardcoded_domain_vocabulary():
    """TC-04: dimensions come from the user's schema, never from the library."""
    root = Path(__file__).resolve().parents[2] / "kse_memory" / "core"
    offenders = []
    for module in ("projection.py", "schema.py"):
        text = (root / module).read_text(encoding="utf-8").lower()
        offenders += [f"{module}:{w}" for w in _DOMAIN_VOCABULARY if w in text]
    assert not offenders, f"hardcoded domain vocabulary in default path: {offenders}"


# --------------------------------------------------------------- model cache
# D-101: the default embedding model resolves from ~/.cache/kse. These tests
# never touch the real home directory — every one redirects it.
def test_cache_dir_defaults_under_home(monkeypatch, tmp_path):
    monkeypatch.delenv("KSE_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert default_cache_dir() == tmp_path / ".cache" / "kse"


def test_cache_dir_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("KSE_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "kse"


def test_cache_dir_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path / "explicit"))
    assert default_cache_dir() == tmp_path / "explicit"


def test_cache_resolution_creates_nothing(monkeypatch, tmp_path):
    """Resolving a path must not have side effects on the filesystem."""
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path / "untouched"))
    default_cache_dir()
    default_model_path("some-model")
    assert not (tmp_path / "untouched").exists()


def test_model_path_is_namespaced_by_model_id(monkeypatch, tmp_path):
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path))
    path = default_model_path("minilm-x")
    assert "minilm-x" in path.parts
    assert tmp_path in path.parents


def test_embedder_defaults_to_the_cache(no_network, monkeypatch, tmp_path):
    """AR-01: with no model cached, the default construction fails loudly."""
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path))
    with pytest.raises(ModelNotAvailableError) as exc:
        OnnxEmbedder()
    assert str(tmp_path) in str(exc.value)


def test_missing_model_error_is_actionable(monkeypatch, tmp_path):
    """The message must say where to put the model, not just that it is absent."""
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path))
    with pytest.raises(ModelNotAvailableError) as exc:
        OnnxEmbedder()
    message = str(exc.value)
    assert "KSE_CACHE_DIR" in message  # names the override
    assert "never downloads" in message  # states the AR-01 guarantee


def test_embedder_accepts_a_cached_model(no_network, monkeypatch, tmp_path):
    """A present model resolves without error and reports its id."""
    monkeypatch.setenv("KSE_CACHE_DIR", str(tmp_path))
    target = default_model_path("onnx-minilm-l6-v2")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not a real onnx graph")
    embedder = OnnxEmbedder()
    assert embedder.model_id == "onnx-minilm-l6-v2"
    assert embedder.model_path == target
