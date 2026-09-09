"""Bind model capabilities to local-process or hosted execution runtimes."""

from __future__ import annotations

from lrauthor.settings import LiteRealitySettings, load_settings


def gen3d_from_settings(settings: LiteRealitySettings | None = None):
    settings = settings or load_settings()
    # Modal is the default runtime, so configured credentials are the whole opt-in; the app name
    # already defaults. Keying off credentials rather than the app name keeps an explicit local
    # GPU runtime reachable now that MODAL_TRELLIS_APP is never empty.
    if settings.modal_configured():
        from lrauthor.models.trellis.modal import ModalTrellisService

        return ModalTrellisService(
            app_name=settings.modal_trellis_app,
            function_name=settings.modal_trellis_function,
            environment_name=settings.modal_environment,
            profile=settings.modal_profile,
            credentials=settings.modal_credentials(),
        )
    if settings.trellis_python:
        from lrauthor.models.trellis.service import LocalTrellisService

        return LocalTrellisService(python=str(settings.trellis_python))
    raise RuntimeError(
        "TRELLIS is not configured: set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET for hosted "
        "execution (the default), or TRELLIS_PYTHON for an explicit local GPU runtime."
    )


def detection_from_settings(settings: LiteRealitySettings | None = None):
    settings = settings or load_settings()
    if settings.modal_configured():
        from lrauthor.models.grounding_dino.modal import ModalDinoService

        return ModalDinoService(
            app_name=settings.modal_dino_app,
            function_name=settings.modal_dino_function,
            environment_name=settings.modal_environment,
            profile=settings.modal_profile,
            credentials=settings.modal_credentials(),
            model_id=settings.dino_model,
            embed_model_id=settings.dino_embed_model,
        )
    if settings.dino_python:
        from lrauthor.models.grounding_dino.service import DinoSubprocessService

        return DinoSubprocessService(python=str(settings.dino_python))
    return None
