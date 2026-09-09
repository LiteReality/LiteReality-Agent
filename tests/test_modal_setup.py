from __future__ import annotations

import pytest

from lrauthor.runtimes import setup


def test_record_profile_creates_env_file(tmp_path):
    env = tmp_path / ".env"

    assert setup.record_profile(env, "team-a") is True
    assert env.read_text(encoding="utf-8") == "MODAL_PROFILE=team-a\n"


def test_record_profile_appends_without_clobbering(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")

    assert setup.record_profile(env, "team-a") is True

    body = env.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-test" in body
    assert "MODAL_PROFILE=team-a" in body


def test_record_profile_rewrites_in_place_on_switch(tmp_path):
    """Switching workspaces must not leave two conflicting assignments behind."""
    env = tmp_path / ".env"
    env.write_text("MODAL_PROFILE=old\nBLENDER_PATH=/b\n", encoding="utf-8")

    assert setup.record_profile(env, "new") is True

    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines == ["MODAL_PROFILE=new", "BLENDER_PATH=/b"]
    assert sum(line.startswith("MODAL_PROFILE=") for line in lines) == 1


def test_record_profile_is_a_noop_when_already_correct(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MODAL_PROFILE=team-a\n", encoding="utf-8")

    assert setup.record_profile(env, "team-a") is False


def test_deploy_reports_a_missing_checkout(tmp_path):
    """Installed-from-wheel users have no deploy/ tree; the error has to say so."""
    with pytest.raises(setup.SetupError, match="source checkout"):
        setup.deploy(tmp_path, "main", "team-a")


def test_deploy_runs_both_apps_with_the_profile_pinned(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs.get("env", {}).get("MODAL_PROFILE")))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(setup, "_run", fake_run)
    setup.deploy(tmp_path, "prod", "team-a")

    assert len(calls) == 2
    for command, profile in calls:
        assert command[1:4] == ["-m", "modal", "deploy"]
        assert command[4:6] == ["--env", "prod"]
        assert profile == "team-a", "each deploy must be pinned to the chosen profile"
    assert str(tmp_path / setup.APPS[0][1]) == calls[0][0][-1]


def test_deploy_raises_when_a_deploy_fails(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    monkeypatch.setattr(setup, "_run", lambda *a, **k: type("R", (), {"returncode": 1})())

    with pytest.raises(setup.SetupError, match="dino"):
        setup.deploy(tmp_path, "main", "team-a")


def test_run_skips_deploy_but_still_records_the_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(setup, "require_client", lambda: None)
    monkeypatch.setattr(
        setup, "deploy", lambda *a, **k: pytest.fail("must not deploy with --skip-deploy")
    )

    assert setup.run(tmp_path, profile="team-a", skip_deploy=True) == 0
    assert "MODAL_PROFILE=team-a" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_tokens_authenticate_without_a_browser_or_profile(tmp_path, monkeypatch):
    """The whole point of the token path: no `modal setup`, no profile, nothing written back."""
    monkeypatch.setattr(setup, "require_client", lambda: None)
    monkeypatch.setattr(
        setup, "authenticate", lambda: pytest.fail("tokens must not trigger a browser login")
    )
    deployed = {}
    monkeypatch.setattr(
        setup,
        "deploy",
        lambda root, env, profile, creds: deployed.update(profile=profile, creds=creds),
    )

    assert setup.run(tmp_path, credentials=("ak-1", "as-2")) == 0
    assert deployed == {"profile": None, "creds": ("ak-1", "as-2")}
    assert not (tmp_path / ".env").exists(), "tokens are already in .env; nothing to write back"


def test_deploy_passes_tokens_to_each_subprocess(tmp_path, monkeypatch):
    for _, relative in setup.APPS:
        app = tmp_path / relative
        app.parent.mkdir(parents=True, exist_ok=True)
        app.write_text("", encoding="utf-8")

    seen = []
    monkeypatch.setattr(
        setup,
        "_run",
        lambda command, **kw: (
            seen.append((kw["env"].get("MODAL_TOKEN_ID"), kw["env"].get("MODAL_TOKEN_SECRET"))),
            type("R", (), {"returncode": 0})(),
        )[1],
    )

    setup.deploy(tmp_path, "main", None, ("ak-1", "as-2"))

    assert seen == [("ak-1", "as-2"), ("ak-1", "as-2")]


def test_settings_need_both_halves_of_the_token_pair():
    """Half a pair cannot authenticate, so it must not read as configured."""
    from lrauthor.settings import LiteRealitySettings

    half = LiteRealitySettings(_env_file=None, MODAL_TOKEN_ID="ak-1")
    assert half.modal_credentials() is None
    assert half.modal_configured() is False

    whole = LiteRealitySettings(_env_file=None, MODAL_TOKEN_ID="ak-1", MODAL_TOKEN_SECRET="as-2")
    assert whole.modal_credentials() == ("ak-1", "as-2")
    assert whole.modal_configured() is True


def test_tokens_reach_subprocesses_through_as_environment():
    from lrauthor.settings import LiteRealitySettings

    settings = LiteRealitySettings(
        _env_file=None, MODAL_TOKEN_ID="ak-1", MODAL_TOKEN_SECRET="as-2"
    )
    env = settings.as_environment()

    assert env["MODAL_TOKEN_ID"] == "ak-1"
    assert env["MODAL_TOKEN_SECRET"] == "as-2"


def test_modal_client_rejects_having_no_credentials_at_all():
    from lrauthor.runtimes.modal import ModalClient

    with pytest.raises(RuntimeError, match="MODAL_TOKEN_ID"):
        ModalClient("app", "fn")


def test_cli_exposes_setup():
    from lrauthor.cli import _parser

    args = _parser().parse_args(["setup", "--skip-deploy", "--profile", "team-a"])

    assert args.profile == "team-a"
    assert args.skip_deploy is True
