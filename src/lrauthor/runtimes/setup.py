"""One-command Modal setup: authenticate, deploy the model apps, record the profile.

Modal is rented compute, not a model API — a token gives you GPUs, not models, so nothing is
callable until the apps are deployed into your own workspace. That is genuinely a one-time step,
but it does not need to be five manual commands. This collapses `modal setup`, profile selection,
both `modal deploy` calls, and the `.env` write into `litereality setup`.

What cannot be collapsed: `modal setup` opens a browser for authentication, and the TRELLIS image
build spends real time and money on an H100. Both are surfaced rather than hidden.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# The TRELLIS image compiles CUDA extensions on an H100. Measured on a fresh workspace 2026-08-06;
# see deploy/modal/README.md for the breakdown. Quoted up front so a first deploy is never a
# surprise charge.
TRELLIS_BUILD_ESTIMATE = "~11 minutes (one-time, per workspace, it will be around 3s in future runs)"

APPS = (("dino", "deploy/modal/dino/app.py"), ("trellis", "deploy/modal/trellis/app.py"))


class SetupError(RuntimeError):
    """A setup step failed in a way the user has to resolve."""


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, **kwargs)


def _modal(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    """Invoke the Modal client through this interpreter so it uses the active env."""
    return _run(
        [sys.executable, "-m", "modal", *args],
        capture_output=capture,
        check=False,
    )


def require_client() -> None:
    """Fail early and actionably when the optional Modal extra is not installed."""
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise SetupError(
            "Modal runtime support is not installed; run `uv sync --extra modal`."
        ) from exc


def authenticated() -> bool:
    """True when the client already has credentials for some profile."""
    result = _modal("profile", "current", capture=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def current_profile() -> str | None:
    result = _modal("profile", "current", capture=True)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def authenticate() -> str:
    """Run `modal setup` (opens a browser) unless credentials already exist, return the profile."""
    if not authenticated():
        print("→ Authenticating with Modal — this opens a browser.")
        if _modal("setup").returncode != 0:
            raise SetupError("`modal setup` failed; re-run it manually to see the error.")
    profile = current_profile()
    if not profile:
        raise SetupError(
            "Modal reports no active profile. Run `uv run modal profile list` and "
            "`uv run modal profile activate <profile>`, then re-run setup."
        )
    return profile


def deploy(
    repo_root: Path,
    environment: str,
    profile: str | None = None,
    credentials: tuple[str, str] | None = None,
) -> None:
    """Deploy both model apps into the workspace the credentials identify."""
    env = {**os.environ}
    if credentials:
        env["MODAL_TOKEN_ID"], env["MODAL_TOKEN_SECRET"] = credentials
    if profile:
        env["MODAL_PROFILE"] = profile
    for label, relative in APPS:
        app_path = repo_root / relative
        if not app_path.is_file():
            raise SetupError(
                f"{relative} not found under {repo_root}. `litereality setup` deploys from a "
                "source checkout; clone the repository and run it from there."
            )
        print(f"→ Deploying {label} ({relative})")
        result = _run(
            [sys.executable, "-m", "modal", "deploy", "--env", environment, str(app_path)],
            env=env,
            check=False,
        )
        if result.returncode != 0:
            raise SetupError(f"deploying {label} failed; see the Modal output above.")


def record_profile(env_path: Path, profile: str) -> bool:
    """Persist MODAL_PROFILE into `.env`. Returns True when the file was changed.

    An existing assignment is rewritten in place rather than appended to, so re-running setup
    after switching workspaces does not leave two conflicting lines behind.
    """
    line = f"MODAL_PROFILE={profile}"
    if not env_path.is_file():
        env_path.write_text(line + "\n", encoding="utf-8")
        return True

    lines = env_path.read_text(encoding="utf-8").splitlines()
    for index, existing in enumerate(lines):
        if existing.strip().startswith("MODAL_PROFILE="):
            if existing.strip() == line:
                return False
            lines[index] = line
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True

    trailing = "" if not lines or lines[-1] == "" else "\n"
    env_path.write_text(env_path.read_text(encoding="utf-8") + trailing + line + "\n", "utf-8")
    return True


def run(
    repo_root: Path,
    *,
    environment: str = "main",
    profile: str | None = None,
    credentials: tuple[str, str] | None = None,
    skip_deploy: bool = False,
) -> int:
    """Authenticate if needed, deploy the apps, record the profile. Returns an exit code."""
    require_client()

    if credentials:
        # Tokens in `.env` are already credentials — there is nothing to log into and nothing to
        # write back, so this path never opens a browser and works unattended.
        print("→ Authenticating with MODAL_TOKEN_ID / MODAL_TOKEN_SECRET from .env")
    else:
        profile = profile or authenticate()
        print(f"→ Modal profile: {profile}")

    if skip_deploy:
        print("→ Skipping deploy (--skip-deploy)")
    else:
        # Common case first: most runs are redeploys against a warm image cache and finish in
        # seconds. Leading with the cold-build number reads as a prediction for this run.
        print("→ Deploying model apps — seconds once the image cache is warm.")
        print(f"  A cold TRELLIS build compiles CUDA extensions on an H100: "
              f"{TRELLIS_BUILD_ESTIMATE}.")
        deploy(repo_root, environment, profile, credentials)

    if profile and not credentials:
        env_path = repo_root / ".env"
        if record_profile(env_path, profile):
            print(f"→ Wrote MODAL_PROFILE to {env_path}")
        else:
            print(f"→ MODAL_PROFILE already set in {env_path}")

    print("\nSetup complete. Verify the whole stack with `uv run python sanity.py`.")
    return 0
