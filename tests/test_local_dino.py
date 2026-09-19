"""Local DINO must override hosted credentials and refuse silent CPU fallback."""

import os
import sys

import pytest

from litereality_agent import SRC_ROOT
from litereality_agent.models.grounding_dino.service import DinoSubprocessService
from litereality_agent.models.registry import detection_from_settings
from litereality_agent.settings import LiteRealitySettings


def test_explicit_local_overrides_modal_credentials():
    settings = LiteRealitySettings(
        _env_file=None, dino_backend="local", dino_python="/local/python",
        dino_model="test-model", modal_token_id="configured", modal_token_secret="configured",
    )
    service = detection_from_settings(settings)
    assert isinstance(service, DinoSubprocessService)
    assert service.python == "/local/python"
    assert service.require_cuda
    assert service.model_id == "test-model"
    assert settings.as_environment()["LR_DINO_BACKEND"] == "local"


def test_local_without_interpreter_does_not_fall_back_to_modal():
    settings = LiteRealitySettings(
        _env_file=None, dino_backend="local", dino_python=None,
        modal_token_id="configured", modal_token_secret="configured",
    )
    with pytest.raises(RuntimeError, match="LR_DINO_PYTHON"):
        detection_from_settings(settings)


@pytest.mark.parametrize("cuda", [True, False])
def test_worker_requires_cuda_and_handles_verbose_stderr(cuda):
    # Synthetic worker exercises the process protocol without torch, a model or a GPU.
    program = f"""
import json, os, sys
sys.stderr.write('model startup log ' * 10000)
sys.stderr.flush()
for line in sys.stdin:
    request = json.loads(line)
    print(json.dumps({{'ok': True, 'cuda': {cuda!r}, 'gpu': 'test GPU',
                      'pythonpath': os.environ['PYTHONPATH']}}), flush=True)
"""
    service = DinoSubprocessService(command=[sys.executable, "-u", "-c", program],
                                    require_cuda=True, env={"PYTHONPATH": "/existing"})
    try:
        if cuda:
            result = service.health()
            assert result["cuda"]
            assert result["pythonpath"].split(os.pathsep) == [str(SRC_ROOT), "/existing"]
            proc = service._proc
            assert service.health() == result
            assert service._proc is proc
        else:
            with pytest.raises(RuntimeError, match="requires CUDA"):
                service.health()
            assert service._proc is None
    finally:
        service.close()
