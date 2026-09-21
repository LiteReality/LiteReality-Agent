import json
import runpy
import sys
from pathlib import Path

from litereality_agent.pipeline.result import StageResult, StageStatus


def test_fresh_smoke_copies_capture_and_records_result(tmp_path, monkeypatch):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "room.usdz").write_bytes(b"source")
    work = tmp_path / "smoke"
    seen = []

    def run(self, context, **options):
        seen.append(context)
        assert context.capture_dir != capture
        assert context.settings.author_provider == "codex"
        assert context.settings.codex_model == "gpt-6-astra"
        return [StageResult("publish", StageStatus.COMPLETED)]

    monkeypatch.setattr("litereality_agent.pipeline.runner.PipelineRunner.run", run)
    monkeypatch.setattr(sys, "argv", ["smoke", str(capture), "--workdir", str(work),
                                      "--settings-from", str(tmp_path)])
    code = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/smoke_support_first.py"))
    assert code["main"]() == 0
    status = json.loads((work / "smoke_status.json").read_text())
    assert status["status"] == "accepted" and status["source_unchanged"]
    assert len(seen) == 1
    monkeypatch.setattr(sys, "argv", ["smoke", str(capture), "--workdir", str(work),
                                      "--settings-from", str(tmp_path), "--resume"])
    assert code["main"]() == 0
    resumed = json.loads((work / "smoke_status.json").read_text())
    assert resumed["history"] == [{k: v for k, v in status.items() if k != "history"}]
    assert resumed["source_unchanged"] and len(seen) == 2
