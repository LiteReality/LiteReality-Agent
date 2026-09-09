#!/usr/bin/env python3
"""Canonical TRELLIS.2 image-to-3D inference and CLI.

Drives ``backends/TRELLIS.2`` (never editing it) to turn a clean object-only
reference PNG — exactly what ``object_init`` produces — into a GLB. The model
**auto-downloads** from HuggingFace (``microsoft/TRELLIS.2-4B``) into
``backends/weights/`` on first run.

    # one image
    python -m lrauthor.models.trellis.inference -i ref.png -o out/ref.glb

    # a whole folder (batch)
    python -m lrauthor.models.trellis.inference -i refs/ -o out/

    # parallel shards (each loads its own ~18 GB model)
    python -m lrauthor.models.trellis.inference -i refs/ -o out/ --parallel 2

Inputs are expected to be object-only PNGs on a (near-)black background, so the
gated RMBG-2.0 matting model is skipped by default (``--no-skip-rembg`` to force
it). Run this module with the configured TRELLIS.2 interpreter; it needs torch 2.6+cu124,
o_voxel, and the other compiled TRELLIS.2 dependencies.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from lrauthor.models import runtime_env as _env

_env.configure_trellis_runtime()

DEFAULT_MODEL = "microsoft/TRELLIS.2-4B"


def prepare_local_model(model: str) -> str:
    """Return a local TRELLIS model dir whose pipeline.json points DINOv3 at a local
    (ungated) copy, so loading never hits the gated ``facebook/dinov3-...`` repo.

    If ``model`` is already a local dir, it's used as-is. For the HF id we ensure the
    snapshot is present, then assemble ``weights/trellis2_4b_local`` (``ckpts`` symlink
    + a patched ``pipeline.json``) and return that — no edits to the HF cache or clone.
    """
    import json

    p = Path(model)
    if p.is_dir() and (p / "pipeline.json").exists():
        return str(p)

    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(model))  # cached under HF_HOME=weights; ckpts already pulled
    dino = _env.resolve_dinov3()
    local = _env.LOCAL_MODEL_DIR
    local.mkdir(parents=True, exist_ok=True)
    ckpts = local / "ckpts"
    if not ckpts.exists():
        ckpts.symlink_to(snap / "ckpts")

    cfg = json.loads((snap / "pipeline.json").read_text())
    icm = cfg.setdefault("args", {}).setdefault("image_cond_model", {}).setdefault("args", {})
    if dino is not None:
        icm["model_name"] = str(dino)
        print(f"[TRELLIS] DINOv3 -> {dino} (local, ungated)")
    else:
        print("[TRELLIS] WARNING: no local DINOv3 found; falling back to the gated HF repo "
              "(set $LITEREALITY_DINOV3 or symlink backends/weights/dinov3, or `huggingface-cli login`).")
    (local / "pipeline.json").write_text(json.dumps(cfg, indent=2))
    return str(local)


def load_pipeline(model_name: str, *, skip_rembg: bool = True):
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if skip_rembg:
        # Must happen before ``from_pretrained`` constructs the matting model.
        install_noop_rembg()
    torch.backends.cudnn.enabled = False
    print(f"[TRELLIS] loading {model_name} (downloads to {_env.WEIGHTS_DIR} on first run)")
    t0 = time.time()
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_name)
    pipeline.cuda()
    print(f"[TRELLIS] model loaded in {time.time() - t0:.1f}s")
    return pipeline


def install_noop_rembg() -> None:
    """Skip the gated ``briaai/RMBG-2.0`` matter; treat near-black bg as transparent."""
    import numpy as np
    from PIL import Image
    from trellis2.pipelines import rembg

    class NoOpRembg:
        def __init__(self, *a, **k):
            pass

        def to(self, device):
            return self

        def cuda(self):
            return self

        def cpu(self):
            return self

        def __call__(self, image):
            rgba = image.convert("RGBA")
            arr = np.array(rgba)
            if arr[:, :, 3].min() < 250:
                return Image.fromarray(arr)
            rgb = arr[:, :, :3]
            mask = (rgb[:, :, 0] > 12) | (rgb[:, :, 1] > 12) | (rgb[:, :, 2] > 12)
            arr[:, :, 3] = mask.astype(np.uint8) * 255
            return Image.fromarray(arr)

    rembg.BiRefNet = NoOpRembg


def generate(pipeline, image_path: Path, output_path: Path, *, seed: int,
             decimation_target: int, texture_size: int, pipeline_type: str | None) -> None:
    import torch
    import o_voxel
    from PIL import Image

    torch.manual_seed(seed)
    image = Image.open(image_path).convert("RGBA")
    mesh = (pipeline.run(image, seed=seed)[0] if pipeline_type is None
            else pipeline.run(image, seed=seed, pipeline_type=pipeline_type)[0])
    mesh.simplify(16777216)  # nvdiffrast limit

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target, texture_size=texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The upstream image installs pillow-simd without libwebp animation support;
    # asking o-voxel for EXT_texture_webp then crashes after generation completes.
    # Standard GLB textures are portable and avoid that private Pillow dependency.
    glb.export(str(output_path), extension_webp=False)


def is_pure_wall_or_floor(stem: str) -> bool:
    """Pure walls (no Door/Window) and Floor are flat planes — TRELLIS makes empty
    meshes for them, so they are skipped (textured separately downstream)."""
    is_pure_wall = stem.startswith("Wall") and "Door" not in stem and "Window" not in stem
    return is_pure_wall or stem == "Floor"


def collect_inputs(input_path: Path, output_path: Path) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        if output_path.suffix.lower() != ".glb":
            output_path = output_path / (input_path.stem + ".glb")
        return [(input_path, output_path)]
    if input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in exts)
        if not images:
            print(f"[TRELLIS] no images in {input_path}", file=sys.stderr)
            sys.exit(2)
        output_path.mkdir(parents=True, exist_ok=True)
        return [(img, output_path / (img.stem + ".glb")) for img in images]
    print(f"[TRELLIS] input not found: {input_path}", file=sys.stderr)
    sys.exit(2)


def _install_stdout_prefix(prefix: str) -> None:
    class _Prefixer:
        def __init__(self, stream):
            self._stream, self._start = stream, True

        def write(self, s):
            for line in s.splitlines(keepends=True):
                if self._start:
                    self._stream.write(prefix)
                    self._start = False
                self._stream.write(line)
                if line.endswith("\n"):
                    self._start = True
            self._stream.flush()

        def flush(self):
            self._stream.flush()

        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout, sys.stderr = _Prefixer(sys.stdout), _Prefixer(sys.stderr)


def _launch_parallel(args: argparse.Namespace) -> int:
    import os
    import subprocess

    N = int(args.parallel)
    base = [sys.executable, os.path.abspath(__file__), "-i", str(args.image), "-o", str(args.output),
            "--model", args.model, "--seed", str(args.seed), "--decimation", str(args.decimation_target),
            "--texture-size", str(args.texture_size)]
    if args.pipeline_type:
        base += ["--pipeline-type", args.pipeline_type]
    if args.skip_existing:
        base.append("--skip-existing")
    if not args.skip_floor_wall:
        base.append("--no-skip-floor-wall")
    if not args.skip_rembg:
        base.append("--no-skip-rembg")
    print(f"[TRELLIS] launching {N} parallel shard(s)", flush=True)
    procs = [subprocess.Popen(base + ["--_shard", f"{i}/{N}"]) for i in range(N)]
    rc = 0
    for p in procs:
        try:
            p.wait()
        except KeyboardInterrupt:
            for q in procs:
                q.terminate()
            raise
        rc = p.returncode or rc
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--image", type=Path, required=True, help="Input PNG or directory (batch).")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output GLB or directory (batch).")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF id or local path (default {DEFAULT_MODEL}).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--decimation", type=int, default=50000, dest="decimation_target", help="Target face count.")
    p.add_argument("--texture-size", type=int, default=4096)
    p.add_argument("--pipeline-type", default=None, help="'512'/'1024'/'1024_cascade'/'1536_cascade'.")
    p.add_argument("--skip-existing", action="store_true", help="Skip if the output GLB already exists.")
    p.add_argument("--no-skip-floor-wall", dest="skip_floor_wall", action="store_false",
                   help="Reconstruct Floor and pure Wall images too (skipped by default).")
    p.set_defaults(skip_floor_wall=True)
    p.add_argument("--parallel", type=int, default=1, help="N subprocess shards, each ~18 GB VRAM.")
    rembg = p.add_mutually_exclusive_group()
    rembg.add_argument("--skip-rembg", dest="skip_rembg", action="store_true",
                       help="Skip gated RMBG-2.0; near-black bg -> alpha (default).")
    rembg.add_argument("--no-skip-rembg", dest="skip_rembg", action="store_false",
                       help="Use the real RMBG-2.0 matter (requires HF access).")
    p.set_defaults(skip_rembg=True)
    p.add_argument("--_shard", default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.parallel > 1 and args._shard is None:
        return _launch_parallel(args)
    if args._shard is not None:
        _install_stdout_prefix(f"[S{args._shard.split('/')[0]}] ")

    pairs = collect_inputs(args.image, args.output)
    if args.skip_floor_wall:
        skipped = [i.name for i, _ in pairs if is_pure_wall_or_floor(i.stem)]
        pairs = [(i, o) for i, o in pairs if not is_pure_wall_or_floor(i.stem)]
        if skipped:
            print(f"[TRELLIS] skipping {len(skipped)} Floor/pure-Wall image(s): {skipped}")
    if args.skip_existing:
        pairs = [(i, o) for i, o in pairs if not o.exists()]
        if not pairs:
            print("[TRELLIS] all outputs already exist, nothing to do.")
            return 0
    if args._shard is not None:
        si, sn = (int(x) for x in args._shard.split("/"))
        pairs = pairs[si::sn]
        if not pairs:
            print("nothing to do")
            return 0

    print(f"[TRELLIS] {len(pairs)} image(s) to process")
    pipeline = load_pipeline(prepare_local_model(args.model), skip_rembg=args.skip_rembg)

    t0 = time.time()
    for idx, (img, out) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {img.name} -> {out.name}")
        ti = time.time()
        try:
            generate(pipeline, img, out, seed=args.seed, decimation_target=args.decimation_target,
                     texture_size=args.texture_size, pipeline_type=args.pipeline_type)
            print(f"  done in {time.time() - ti:.1f}s  ({out.stat().st_size / 1024:.0f} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}", file=sys.stderr)
    print(f"[TRELLIS] all done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
