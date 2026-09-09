from __future__ import annotations

from pathlib import Path


def test_service_result_is_an_error_when_an_expected_glb_is_missing(tmp_path, monkeypatch):
    from litereality_agent.pipeline.author.reconstruct import flow, service

    reference = tmp_path / "Chair0.png"
    reference.write_bytes(b"reference")
    output = tmp_path / "reconstructed"

    class EmptyService:
        name = "empty"

        def reconstruct_many(self, *_args, **_kwargs):
            return {"Chair0": str(output / "Chair0.glb")}

    monkeypatch.setattr(flow.config, "reconstruct_dir", lambda _scan: output)
    monkeypatch.setattr(flow, "collect_references", lambda _scan: [("Chair0", reference)])
    monkeypatch.setattr(service, "_SERVICE", EmptyService())

    result = flow.run_for_scan("scan")

    assert result["status"] == "failed"
    assert result["generated"] == 0
    assert result["failed_assets"] == ["Chair0"]


def test_flow_returns_nonzero_when_requested_assets_are_missing(monkeypatch):
    from litereality_agent.pipeline.measure import flow

    monkeypatch.setattr(flow, "resolve_scan", lambda *_args: (Path("capture"), "scan"))
    monkeypatch.setattr(flow, "process_scan", lambda *_args: {"scan": "scan"})
    monkeypatch.setattr(flow, "summarize", lambda _results: None)
    monkeypatch.setattr(flow, "_expected_assets", lambda _scan: ["Chair0"])
    monkeypatch.setattr(flow, "_missing_assets", lambda _scan, _expected: ["Chair0"])

    assert flow.main(["--scan", "capture", "--reconstruct"]) == 1


def test_flow_allows_a_room_with_no_expected_assets(monkeypatch):
    from litereality_agent.pipeline.measure import flow

    monkeypatch.setattr(flow, "resolve_scan", lambda *_args: (Path("capture"), "scan"))
    monkeypatch.setattr(flow, "process_scan", lambda *_args: {"scan": "scan"})
    monkeypatch.setattr(flow, "summarize", lambda _results: None)
    monkeypatch.setattr(flow, "_expected_assets", lambda _scan: [])
    monkeypatch.setattr(flow, "_missing_assets", lambda _scan, _expected: [])

    assert flow.main(["--scan", "capture", "--reconstruct"]) == 0
