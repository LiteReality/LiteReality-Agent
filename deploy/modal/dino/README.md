# DINO on Modal

This scale-to-zero app serves both GroundingDINO detection and DINOv2 embeddings through the
canonical model code. Its image is CPU-built from precompiled wheels; an L4 is attached only while
requests run. Public Hugging Face weights are cached in the `litereality-dino-models` Volume.

Deploy into whichever workspace the active profile names:

```bash
MODAL_PROFILE=<profile> modal deploy --env main deploy/modal/app.py  (deploys trellis and dino together)
```

Configure the pipeline by setting `MODAL_PROFILE` to that same profile; `MODAL_DINO_APP`,
`MODAL_DINO_FUNCTION`, and `MODAL_ENVIRONMENT` already default. No Hugging Face token is needed —
both models are public.
