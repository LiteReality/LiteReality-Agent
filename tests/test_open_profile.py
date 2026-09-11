"""The `open` profile: a brief with the evidence pack, its own budgets, and the codex harness by default."""

from __future__ import annotations

from litereality_agent.agent import author


def test_open_is_a_registered_profile_with_larger_budgets():
    assert "open" in author.PROFILES
    assert author.OPEN_STEP_BUDGET > 100 and author.OPEN_MAX_TURNS > 140


def test_the_open_brief_points_at_the_pack_and_the_gate():
    text = author.OPEN_PROMPT.format(
        stitch_lines="", scan="/s", surface_list="Wall0", rhythm="", evidence="/a/evidence",
        n_frames=59, python="/py", blender="/bl", scratch="/sc")
    assert "/a/evidence/helpers" in text and "measure.probe_depth" in text
    assert "litereality_agent.pipeline.room_qc.validate" in text and "/py -m" in text
    assert "rests_on=" in text and "fetch_material" in text
    # freedom, stated: the process is the model's, the result is gated
    assert "HOW YOU WORK IS YOURS" in text
    assert "EDIT `Room.py` AT LEAST ONCE EVERY" not in text


def test_the_scripted_profiles_still_format_with_the_extra_fields():
    for name in ("base", "detail", "simulation"):
        author.PROFILES[name].format(stitch_lines="", scan="/s", surface_list="Wall0", rhythm=author.RHYTHM,
                                     evidence="", n_frames=0, python="", blender="", scratch="")


def test_entrypoint_defaults_budgets_per_profile(monkeypatch):
    import sys

    from litereality_agent.pipeline.realism_authoring.author import entrypoint

    seen = {}

    async def fake_run(room, surface_ref, scan, model, max_turns, profile, **kw):
        seen.update(max_turns=max_turns, profile=profile, **kw)
        return 0

    monkeypatch.setattr(entrypoint, "run", fake_run)
    monkeypatch.delenv("AUTHOR_STEPS", raising=False)
    for profile, turns, steps in (("open", author.OPEN_MAX_TURNS, author.OPEN_STEP_BUDGET), ("simulation", 140, 100)):
        monkeypatch.setattr(sys, "argv", ["x", "--room", "/r", "--surface-ref", "/s", "--scan", "/c", "--profile", profile])
        try:
            entrypoint.main()
        except SystemExit:
            pass
        assert (seen["max_turns"], seen["step_budget"]) == (turns, steps), profile
