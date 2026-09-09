# articulated-glb-agent

A fully automatic batch tool: reference image → an articulated (pull-out, openable) GLB model.

Built on the Claude Agent SDK. Each image gets its own agent session, which analyses the picture,
downloads PBR textures, writes and runs a Blender build script, validates the GLB structure, and
renders open/closed previews to check its own work — iterating until the result matches the
reference.

```
articulated-glb-agent/
├── batch_glb.py            # batch entry point (the only file you run)
├── requirements.txt
├── README.md
└── .claude/skills/image-to-articulated-glb/   # the skill, shipped with the package
    ├── SKILL.md            # the pipeline description (this is what the agent reads)
    └── scripts/            # reusable scripts: blender_lib / fetch_polyhaven / validate_glb /
                            # render_glb_preview / make_selfcontained ...
```

## Requirements

| dependency | notes |
|---|---|
| Python ≥ 3.10 | `pip install -r requirements.txt`. claude-agent-sdk bundles the Claude CLI, so Claude Code need not be installed separately. |
| Blender 4.x/5.x | Looks for `/opt/homebrew/bin/blender` first, otherwise `blender` on PATH. macOS: `brew install blender`. On Linux, just put blender on PATH. |
| ANTHROPIC_API_KEY | A Claude API key, billed per token. Without it, falls back to the Claude Code credentials already logged in on this machine. |
| network | Textures are downloaded from Poly Haven; on failure it switches to procedural textures automatically. |

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# a single image
python3 batch_glb.py /path/to/chair.png

# a whole directory: 2 at a time, one retry on failure, skipping what is already done
python3 batch_glb.py /path/to/images/ --concurrency 2 --retries 1 --skip-existing

# see which jobs would run, without spending anything
python3 batch_glb.py /path/to/images/ --dry-run
```

## Output

By default, alongside the image: `<name>_glb/<name>.glb`, containing:

- `<name>.glb` — textures and animation embedded. Every moving part carries glTF node extras
  (`articulation_type` prismatic/revolute, `articulation_axis`, `limit_min`/`limit_max`), which a
  simulator can read directly.
- `object.py` — a SELF-CONTAINED Blender script. The helper library is inlined, so this single file
  rebuilds the object with no external dependencies: `blender -b --python object.py` (reads
  `./textures` and writes `./<name>.glb` by default).
- `object.md` — how to edit this object: the constants you can change, what the joints do, and the
  rebuild command.
- `previews/preview_closed.png`, `preview_open.png` — the renders used for self-checking.
- `textures/` and the build script — kept so the result can be reviewed or rebuilt.

`--out-base ./out` collects everything under `./out/<name>/` instead.

On completion it prints a summary (succeeded / failed / total cost) and writes
`batch_glb_report.json` with the cost, elapsed time, retry count and any error, per image.

## Options

| option | default | notes |
|---|---|---|
| `--concurrency` | 2 | how many jobs run in parallel. Blender rendering is CPU-hungry, so going much higher is not recommended. |
| `--retries` | 1 | retries when validation fails — a missing GLB, or missing previews. |
| `--max-budget-usd` | 5.0 | cost ceiling per attempt, enforced by the SDK. This is only a real limit on metered API access; on a logged-in Claude CLI the figure is notional. |
| `--max-turns` | 80 | maximum agent turns per attempt. |
| `--model` | SDK default | model override. |
| `--skip-existing` | off | skip images whose output is already complete — useful for resuming. |

## Cost and time

Roughly $1–3 and 3–8 minutes per image, depending on how complex the object is and how many
self-check iterations it needs. Run one or two before a large batch to confirm the per-image cost.

## Security notes

- Pass the API key through the environment only. Never write it into code or commit it to git.
- The agent runs Bash and writes files automatically under `bypassPermissions`, which the pipeline
  requires. Run it in a dedicated directory or on a dedicated machine, and never point it at a
  directory holding sensitive data.
