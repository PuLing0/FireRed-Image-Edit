"""Configuration for FireRed Agent."""

import os
from typing import Any


def _get_first_env(*names: str, default: str = "") -> str:
    """Return the first non-empty environment variable from *names*."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


def get_gemini_api_key() -> str:
    """Return the Gemini API key."""
    return _get_first_env("GEMINI_API_KEY", "GOOGLE_API_KEY")


def get_gemini_model_name() -> str:
    """Return the Gemini model name."""
    return _get_first_env("GEMINI_MODEL_NAME", default="gemini-2.5-flash")


def get_gemini_base_url() -> str:
    """Return the optional Gemini-compatible base URL / API endpoint."""
    return _get_first_env(
        "GEMINI_BASE_URL",
        "GEMINI_API_BASE_URL",
        "GEMINI_API_ENDPOINT",
    ).rstrip("/")


def get_gemini_config_kwargs() -> dict[str, Any]:
    """Build kwargs for ``google.generativeai.configure``."""
    kwargs: dict[str, Any] = {}

    api_key = get_gemini_api_key()
    if api_key:
        kwargs["api_key"] = api_key

    base_url = get_gemini_base_url()
    if base_url:
        # Custom Gemini-compatible gateways are typically REST-only.
        kwargs["transport"] = "rest"
        kwargs["client_options"] = {"api_endpoint": base_url}

    return kwargs


# Backward-compatible module-level aliases
GEMINI_API_KEY: str = get_gemini_api_key()
GEMINI_MODEL_NAME: str = get_gemini_model_name()
GEMINI_BASE_URL: str = get_gemini_base_url()

# ---------------------------------------------------------------------------
# Image stitching defaults
# ---------------------------------------------------------------------------
# Target total area (pixels) for the stitched output canvas
STITCH_TARGET_AREA: int = 1024 * 1024  # ~1 mega-pixel
# Max number of output images that FireRed-Image-Edit accepts
MAX_OUTPUT_IMAGES: int = 3
# Min / max aspect ratio for the stitched canvas
STITCH_MIN_ASPECT: float = 0.5   # portrait  (H = 2W)
STITCH_MAX_ASPECT: float = 2.0   # landscape (W = 2H)
# Padding colour when a small gap remains
STITCH_PAD_COLOR: tuple[int, int, int] = (255, 255, 255)

# ---------------------------------------------------------------------------
# Recaption
# ---------------------------------------------------------------------------
RECAPTION_TARGET_LENGTH: int = 512  # target word/character count
