"""The supported pipeline, in execution order."""

from lrauthor.pipeline.author import realism, reconstruct, seed
from lrauthor.pipeline.compile import publish, simulate
from lrauthor.pipeline.measure import ingest
from lrauthor.pipeline.stage import Stage

STAGES = (
    Stage("ingest", ingest.run, is_complete=ingest.complete),
    Stage("reconstruct", reconstruct.run, ("ingest",), is_complete=reconstruct.complete),
    Stage("seed", seed.run, ("reconstruct",), is_complete=seed.complete),
    Stage("author", realism.run, ("seed",), is_complete=realism.complete),
    Stage("publish", publish.run, ("author",), is_complete=publish.complete),
    # LAST, and not required. Everything it reads — the shell, the layout, the built glb and each
    # object's own compiled physics — exists by now, and a room that never reaches a simulator is
    # still a finished room. Making it required would fail a publish over an optional export.
    Stage("simulate", simulate.run, ("publish",), required=False,
          is_complete=simulate.complete),
)

__all__ = ["STAGES"]
