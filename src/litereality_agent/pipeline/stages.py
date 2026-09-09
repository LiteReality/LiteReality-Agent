"""The supported pipeline, in execution order."""

from litereality_agent.pipeline import simulate
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
    # LAST, and not required. Everything it reads — the shell, the layout, the built glb and each
    # object's own compiled physics — exists by now, and a room that never reaches a simulator is
    # still a finished room. Making it required would fail a publish over an optional export.
    Stage("simulate", simulate.run, ("publish",), required=False,
          is_complete=simulate.complete),
)

__all__ = ["STAGES"]
