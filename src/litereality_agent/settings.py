"""Typed application settings loaded once at the composition boundary."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from litereality_agent import REPO_ROOT


class LiteRealitySettings(BaseSettings):
    """Environment contract for the pipeline and its subprocess adapters.

    Pydantic's source ordering gives process environment priority over dotenv
    files. ``load`` supplies ``models.env`` first and ``.env`` second, so user
    machine/secrets configuration overrides repository model defaults.
    """

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    repo_root: Path = Field(default=REPO_ROOT, validation_alias="LR_REPO_ROOT")
    scans_dir: Path | None = Field(default=None, validation_alias="LR_SCANS_DIR")
    output_root: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LITEREALITY_OUTPUT", "OBJECT_INIT_OUTPUT"),
    )
    final_root: Path | None = Field(default=None, validation_alias="LITEREALITY_FINAL")
    scene: Path | None = Field(default=None, validation_alias="LR_SCENE")
    authoring_root: Path | None = Field(default=None, validation_alias="LR_AUTHORING")
    studio_keys: Path | None = Field(default=None, validation_alias="LR_STUDIO_KEYS")

    blender: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LITEREALITY_BLENDER", "BLENDER_PATH", "BLENDER"),
    )
    dino_python: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LR_DINO_PYTHON", "GROUNDING_DINO_PYTHON"),
    )
    trellis_python: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LITEREALITY_TRELLIS_PYTHON", "TRELLIS_PYTHON"),
    )
    procedural_python: Path | None = Field(
        default=None, validation_alias="LITEREALITY_PROCEDURAL_PYTHON"
    )

    # WHICH coding agent drives the file-editing sessions. Independent of the model knobs below:
    # the harness is the agent loop, the model is the brain inside it. Per-role overrides default
    # to `agent_provider` in `resolve_dependent_defaults`.
    agent_provider: str = Field(default="claude", validation_alias="LR_AGENT_PROVIDER")
    author_provider: str | None = Field(default=None, validation_alias="LR_AUTHOR_PROVIDER")
    materials_provider: str | None = Field(default=None, validation_alias="LR_MATERIALS_PROVIDER")
    quality_provider: str | None = Field(default=None, validation_alias="LR_QUALITY_PROVIDER")
    refine_provider: str | None = Field(default=None, validation_alias="LR_REFINE_PROVIDER")
    procedural_provider: str | None = Field(
        default=None, validation_alias="LR_PROCEDURAL_PROVIDER"
    )
    codex_model: str | None = Field(default=None, validation_alias="LR_CODEX_MODEL")
    codex_effort: str = Field(default="high", validation_alias="LR_CODEX_EFFORT")

    harness_model: str = Field(default="claude-opus-5", validation_alias="HARNESS_MODEL")
    critic_model: str | None = Field(default=None, validation_alias="HARNESS_CRITIC_MODEL")
    chair_judge_model: str = Field(
        default="claude-opus-5", validation_alias="LR_CHAIR_JUDGE_MODEL"
    )
    procedural_model: str = Field(
        default="claude-opus-5", validation_alias="LR_PROCEDURAL_MODEL"
    )
    image_model: str = Field(default="gpt-image-2", validation_alias="LR_OPENAI_IMAGE_MODEL")
    dino_model: str = Field(
        default="IDEA-Research/grounding-dino-tiny", validation_alias="LR_DINO_MODEL"
    )
    dino_embed_model: str = Field(
        default="facebook/dinov2-small", validation_alias="LR_DINO_EMBED_MODEL"
    )

    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    # Which backend generates reference images. `models.env` documents it and `image_gen` reads it
    # off the environment, but nothing carried it from `.env` INTO the subprocess that does the
    # generating — so the documented Gemini path failed on a missing OpenAI key and fell back.
    image_provider: str | None = Field(default=None, validation_alias="LR_IMAGE_PROVIDER")
    # Modal is the default execution runtime, so the app names carry the deployed defaults and
    # MODAL_PROFILE alone selects hosted execution. Override these only when a workspace deploys
    # the apps under different names.
    modal_trellis_app: str = Field(
        default="litereality", validation_alias="MODAL_TRELLIS_APP"
    )
    modal_trellis_function: str = Field(
        default="trellis", validation_alias="MODAL_TRELLIS_FUNCTION"
    )
    modal_dino_app: str = Field(default="litereality", validation_alias="MODAL_DINO_APP")
    modal_dino_function: str = Field(default="dino", validation_alias="MODAL_DINO_FUNCTION")
    modal_environment: str = Field(default="main", validation_alias="MODAL_ENVIRONMENT")
    modal_profile: str | None = Field(default=None, validation_alias="MODAL_PROFILE")
    # Modal authenticates from either a token pair or a named profile in ~/.modal.toml. The token
    # pair is what a fresh clone should use: it is two values pasted into `.env` like any other
    # key, needs no browser, and leaves nothing on the machine outside the checkout.
    modal_token_id: SecretStr | None = Field(default=None, validation_alias="MODAL_TOKEN_ID")
    modal_token_secret: SecretStr | None = Field(
        default=None, validation_alias="MODAL_TOKEN_SECRET"
    )

    AGENT_PROVIDERS: ClassVar[tuple[str, ...]] = ("claude", "codex")
    _PROVIDER_ROLES: ClassVar[tuple[str, ...]] = (
        "author_provider",
        "materials_provider",
        "quality_provider",
        "refine_provider",
        "procedural_provider",
    )

    @model_validator(mode="after")
    def resolve_dependent_defaults(self) -> "LiteRealitySettings":
        if not self.critic_model or self.critic_model == "$HARNESS_MODEL":
            self.critic_model = self.harness_model
        # An unknown harness name must fail HERE, at the composition boundary, rather than deep
        # inside a paid session: `LR_AGENT_PROVIDER=codexx` would otherwise fall through to the
        # claude default and silently run the wrong agent.
        self.agent_provider = self._checked_provider(self.agent_provider) or "claude"
        for role in self._PROVIDER_ROLES:
            resolved = self._checked_provider(getattr(self, role)) or self.agent_provider
            setattr(self, role, resolved)
        return self

    @classmethod
    def _checked_provider(cls, value: str | None) -> str | None:
        chosen = str(value or "").strip().lower()
        # `models.env` is read by python-dotenv, which leaves an unexpanded `${VAR:-$OTHER}`
        # default as the literal `$other`. Treat that as unset rather than as a bad provider —
        # `critic_model` has the same shape of workaround above.
        if not chosen or chosen.startswith("$"):
            return None
        if chosen not in cls.AGENT_PROVIDERS:
            raise ValueError(
                f"unsupported agent provider {chosen!r} "
                f"(expected one of {', '.join(cls.AGENT_PROVIDERS)})"
            )
        return chosen

    @classmethod
    def load(cls, root: Path | None = None, **overrides: Any) -> "LiteRealitySettings":
        root = (root or REPO_ROOT).resolve()
        return cls(
            _env_file=(root / "models.env", root / ".env"),
            _env_file_encoding="utf-8",
            **overrides,
        )

    def resolved_output_root(self) -> Path:
        return (self.output_root or self.final_root or (self.repo_root / "run")).resolve()

    def resolved_scans_dir(self) -> Path:
        return (self.scans_dir or (self.repo_root / "scans_uploaded")).resolve()

    def modal_credentials(self) -> tuple[str, str] | None:
        """The `(token_id, token_secret)` pair, or None unless BOTH are set.

        Half a token pair cannot authenticate, so it must not read as configured — otherwise the
        runtime selects Modal and fails at the call instead of falling back or saying what is
        missing.
        """
        if self.modal_token_id and self.modal_token_secret:
            return (
                self.modal_token_id.get_secret_value(),
                self.modal_token_secret.get_secret_value(),
            )
        return None

    def modal_configured(self) -> bool:
        """Whether hosted execution is selected — the one opt-in the runtime keys off.

        Either authentication route counts: a token pair in `.env`, or a named profile in
        `~/.modal.toml`.
        """
        return bool(self.modal_profile) or self.modal_credentials() is not None

    def as_environment(self) -> dict[str, str]:
        """Serialize canonical names for subprocesses that have not yet been typed."""
        values: dict[str, object | None] = {
            "LR_REPO_ROOT": self.repo_root,
            "LR_SCANS_DIR": self.resolved_scans_dir(),
            "LITEREALITY_OUTPUT": self.resolved_output_root(),
            "LITEREALITY_FINAL": self.final_root or self.resolved_output_root(),
            "LR_SCENE": self.scene,
            "LR_AUTHORING": self.authoring_root,
            "LR_STUDIO_KEYS": self.studio_keys,
            "LITEREALITY_BLENDER": self.blender,
            "LR_DINO_PYTHON": self.dino_python,
            "LITEREALITY_TRELLIS_PYTHON": self.trellis_python,
            "LITEREALITY_PROCEDURAL_PYTHON": self.procedural_python,
            "LR_AGENT_PROVIDER": self.agent_provider,
            "LR_AUTHOR_PROVIDER": self.author_provider,
            "LR_MATERIALS_PROVIDER": self.materials_provider,
            "LR_QUALITY_PROVIDER": self.quality_provider,
            "LR_REFINE_PROVIDER": self.refine_provider,
            "LR_PROCEDURAL_PROVIDER": self.procedural_provider,
            "LR_CODEX_MODEL": self.codex_model,
            "LR_CODEX_EFFORT": self.codex_effort,
            "HARNESS_MODEL": self.harness_model,
            "HARNESS_CRITIC_MODEL": self.critic_model,
            "LR_CHAIR_JUDGE_MODEL": self.chair_judge_model,
            "LR_PROCEDURAL_MODEL": self.procedural_model,
            "LR_OPENAI_IMAGE_MODEL": self.image_model,
            "LR_DINO_MODEL": self.dino_model,
            "LR_DINO_EMBED_MODEL": self.dino_embed_model,
            "MODAL_TRELLIS_APP": self.modal_trellis_app,
            "MODAL_TRELLIS_FUNCTION": self.modal_trellis_function,
            "MODAL_DINO_APP": self.modal_dino_app,
            "MODAL_DINO_FUNCTION": self.modal_dino_function,
            "MODAL_ENVIRONMENT": self.modal_environment,
        }
        values["LR_IMAGE_PROVIDER"] = self.image_provider
        secrets = {
            "OPENAI_API_KEY": self.openai_api_key,
            "GEMINI_API_KEY": self.gemini_api_key,
            "ANTHROPIC_API_KEY": self.anthropic_api_key,
            # Exported so `modal deploy` and the model subprocesses authenticate from `.env`
            # without a ~/.modal.toml on the machine.
            "MODAL_TOKEN_ID": self.modal_token_id,
            "MODAL_TOKEN_SECRET": self.modal_token_secret,
        }
        env = {key: str(value) for key, value in values.items() if value is not None}
        env.update(
            {key: value.get_secret_value() for key, value in secrets.items() if value is not None}
        )
        return env

    def apply_environment(self) -> None:
        """Populate canonical variables without overriding the invoking shell."""
        for key, value in self.as_environment().items():
            os.environ.setdefault(key, value)


def load_settings(root: Path | None = None, **overrides: Any) -> LiteRealitySettings:
    return LiteRealitySettings.load(root, **overrides)
