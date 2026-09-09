# TRELLIS on Modal

This is a deployment wrapper around
`lrauthor.models.trellis.inference`; model inference remains in `src/` and the Modal
runtime only supplies a GPU, parallel execution, and a persistent weight cache.

Modal associates a deployment with the workspace belonging to the active profile; the workspace
cannot be selected by app source code. Confirm which one you are about to bill, then deploy into
its `main` environment:

```bash
modal profile list
modal profile activate <name>
modal profile current

modal deploy --env main deploy/modal/trellis/app.py
```

## No Hugging Face account is needed

TRELLIS.2's image encoder is the gated `facebook/dinov3-vitl16-pretrain-lvd1689m`, which is why
this app previously required a `huggingface` Modal secret. It no longer does. The DINOv3 licence
grants the right to "distribute, copy" the trained weights to third parties provided a copy of the
licence travels with them, so the image bakes in an ungated mirror instead:

- repo `camenduru/dinov3-vitl16-pretrain-lvd1689m`
- revision `3c276edd87d6f6e569ff0c4400e086807d0f3881`
- `model.safetensors` sha256 `dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`

The build verifies that digest and fails if it does not match, so a mirror force-push cannot
silently swap the weights. Four independent uploaders publish byte-identical copies under that same
digest, which is how the weights were confirmed unmodified without holding gate access. Always pin
by revision. `resolve_dinov3()` finds the baked copy through `LITEREALITY_DINOV3` and repoints
`pipeline.json` at it, so loading never reaches the gated repo.

`microsoft/TRELLIS.2-4B` is MIT and ungated, so no token is needed for it either.

The base CUDA image, OS packages, PyTorch, and Python dependencies build on CPU. Modal attaches an
H100 only to the separately cached native-extension layer required by the upstream TRELLIS
installer; the deployed inference function also uses an H100. Source-only application changes do
not rebuild the extension layer. The function is intentionally limited to one container, performs
zero application retries, and scales that container down after ten idle seconds to bound spend.

The weights are hosted on Hugging Face, but the model is not currently served by a Hugging Face
Inference Provider. The first invocation downloads `microsoft/TRELLIS.2-4B` into the
`litereality-trellis-models` Volume; later containers reuse that cache. Note that this download
runs inside the `gpu="H100"` function, so several GB transfer at H100 prices — see the known
inefficiencies in `../README.md`.

Configure the application by setting `MODAL_PROFILE` to your profile name; the app name, function
name, and environment already default. Install client support with `uv sync --extra modal`.

Measured on a fresh workspace, 2026-08-06: the image built in 11.3 minutes for about $0.45, split
almost evenly in time between the CPU dependency layers (318.0s) and the H100 extension compile
(318.4s), with the GPU layer carrying roughly 78% of the cost. Redeploy after a source-only change
took 2.8 seconds. The first request returned a valid 512-texture GLB from `ChairCluster0` — 33,898
vertices, 47,793 faces — in 295.7s including the weights download; later requests skip it. Neither
building nor inference is part of the offline test suite.
