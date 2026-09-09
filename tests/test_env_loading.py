"""`.env` + `models.env` must mean the same thing from the CLI as from the CLI.

`models.env` is shell, so shell entry points just source it. Everything reachable only through
the CLI could not, and the two silently diverged: `./the CLI` generated references with the model
the file names while `uv run -m lrauthor scene_init` fell back to the code default. Nothing
fails — you just get different images, and only notice by inspecting them.

The precedence chain is the load-bearing part: shell env > `.env` > `models.env`, matching the
order the CLI sources them and the `${K:-default}` semantics of the file itself.
"""

from __future__ import annotations

import pytest

from lrauthor.settings import load_settings

MODELS_ENV = """\
# the one place models are picked
export HARNESS_MODEL="${HARNESS_MODEL:-claude-opus-5}"
export HARNESS_CRITIC_MODEL="${HARNESS_CRITIC_MODEL:-$HARNESS_MODEL}"
export LR_OPENAI_IMAGE_MODEL="${LR_OPENAI_IMAGE_MODEL:-gpt-image-2}"
export LR_DINO_MODEL="${LR_DINO_MODEL:-IDEA-Research/grounding-dino-tiny}"   # trailing comment
PLAIN_KEY=plain-value
export LR_UNRESOLVED="${LR_UNRESOLVED:-$NOT_SET_ANYWHERE}"
"""

KEYS = ["HARNESS_MODEL", "HARNESS_CRITIC_MODEL", "LR_OPENAI_IMAGE_MODEL", "LR_DINO_MODEL",
        "PLAIN_KEY", "LR_UNRESOLVED", "OPENAI_API_KEY"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "models.env").write_text(MODELS_ENV, encoding="utf-8")
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_models_env_is_loaded(repo):
    settings = load_settings(repo)
    assert settings.image_model == "gpt-image-2"
    assert settings.harness_model == "claude-opus-5"


def test_trailing_comment_is_not_part_of_the_value(repo):
    """`export K="${K:-v}"   # note` — matching the substitution against the whole tail fails on
    these, and the literal `${...}` becomes the model name."""
    assert load_settings(repo).dino_model == "IDEA-Research/grounding-dino-tiny"


def test_a_default_may_name_another_variable(repo):
    assert load_settings(repo).critic_model == "claude-opus-5"


def test_unresolvable_reference_sets_nothing(repo):
    """Better unset than the literal string `$NOT_SET_ANYWHERE` reaching an API as a model id."""
    assert "LR_UNRESOLVED" not in load_settings(repo).as_environment()


def test_unknown_assignments_are_ignored(repo):
    assert "PLAIN_KEY" not in load_settings(repo).as_environment()


def test_shell_env_wins(repo, monkeypatch):
    monkeypatch.setenv("LR_OPENAI_IMAGE_MODEL", "gpt-image-1")
    assert load_settings(repo).image_model == "gpt-image-1"


def test_dotenv_wins_over_models_env(repo):
    """`.env` is yours; `models.env` only ever supplies defaults — the same order the CLI uses."""
    (repo / ".env").write_text("LR_OPENAI_IMAGE_MODEL=gpt-image-1\n", encoding="utf-8")
    assert load_settings(repo).image_model == "gpt-image-1"


def test_empty_template_values_are_ignored(repo):
    """`.env.example` ships `OPENAI_API_KEY=` — an empty value must not shadow the real one."""
    (repo / ".env").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
    settings = load_settings(repo, openai_api_key="sk-real")
    assert settings.openai_api_key.get_secret_value() == "sk-real"


def test_loading_twice_is_stable(repo):
    first = load_settings(repo)
    second = load_settings(repo)
    assert second == first


# --- reference artifact names ------------------------------------------------ #
def test_artifact_names_are_provider_neutral():
    """The two reference images are named for WHAT they are, not who made them — the old
    `gemini_input` / `nano_banana_raw` outlived both providers."""
    from lrauthor.pipeline.measure import paths as oi

    assert oi.INPUT_SHEET == "input2imagegen.jpg"
    assert oi.CLEAN_REFERENCE == "clean_obj_reference.png"


def test_readers_accept_the_pre_rename_spelling(tmp_path):
    """An existing tree cost ~6 minutes of model calls to produce. Renaming the writers must not
    make it unreadable — `ref_artifact` finds either spelling, newest name first."""
    from lrauthor.pipeline.measure import paths as oi

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "gemini_input.jpg").write_bytes(b"")
    (legacy / "nano_banana_raw.png").write_bytes(b"")
    assert oi.ref_artifact(legacy, oi.INPUT_SHEET).name == "gemini_input.jpg"
    assert oi.ref_artifact(legacy, oi.CLEAN_REFERENCE).name == "nano_banana_raw.png"

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / oi.INPUT_SHEET).write_bytes(b"")
    assert oi.ref_artifact(fresh, oi.INPUT_SHEET).name == oi.INPUT_SHEET
    assert oi.ref_artifact(fresh, oi.CLEAN_REFERENCE) is None  # absent, not guessed


def test_writers_never_emit_the_old_names():
    """A write site left behind would split every reference dir across two naming schemes."""
    import re
    from pathlib import Path as P

    pkg = P(__file__).resolve().parents[1] / "src" / "lrauthor"
    write = re.compile(r"""(?:out_dir|d|od|croot|ref_dir)\s*/\s*["'](?:gemini_input|nano_banana_raw)""")
    offenders = [
        f"{p.relative_to(pkg)}:{i}"
        for p in pkg.rglob("*.py")
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
        if write.search(line) and " or " not in line  # `or <old>` is a read fallback
    ]
    assert not offenders, f"old artifact name written at {offenders}"
