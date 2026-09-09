"""The supported pipeline, in execution order."""

from litereality_agent.pipeline.realism_authoring import author
from litereality_agent.pipeline.room_qc import publish
from litereality_agent.pipeline.scene_init import ingest, reconstruct, seed
from litereality_agent.pipeline.stage import Stage

STAGES = (
    Stage("ingest", ingest.run, is_complete=ingest.complete),
    Stage("reconstruct", reconstruct.run, ("ingest",), is_complete=reconstruct.complete),
    Stage("seed", seed.run, ("reconstruct",), is_complete=seed.complete),
    Stage("author", author.run, ("seed",), is_complete=author.complete),
    Stage("publish", publish.run, ("author",), is_complete=publish.complete),
)

__all__ = ["STAGES"]
