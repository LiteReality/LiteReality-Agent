"""Composable LiteReality pipeline.

The public API is deliberately small: build a :class:`RunContext`, then ask the
runner to execute named stages.  Stage implementations own domain work; this
package owns ordering, reuse, failure policy, and reporting.
"""

from lrauthor.pipeline.context import RunContext
from lrauthor.pipeline.result import StageResult, StageStatus
from lrauthor.pipeline.runner import PIPELINE, PipelineRunner

__all__ = ["PIPELINE", "PipelineRunner", "RunContext", "StageResult", "StageStatus"]
