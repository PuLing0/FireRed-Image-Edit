"""Gemini-backed visual helper tools for the comprehensive agent."""

from __future__ import annotations

import json
import re
import traceback
import warnings
from typing import Any

from PIL import Image

from agent.config import get_gemini_config_kwargs, get_gemini_model_name


def _import_genai():
    """Lazily import ``google.generativeai``."""
    try:
        import google.generativeai as _genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai is required for Gemini-backed visual tools. "
            "Install it with: pip install google-generativeai"
        ) from exc
    return _genai


def _init_gemini() -> Any:
    """Return a low-temperature Gemini model."""
    genai = _import_genai()
    genai.configure(**get_gemini_config_kwargs())
    return genai.GenerativeModel(
        model_name=get_gemini_model_name(),
        generation_config={
            "temperature": 0.1,
            "max_output_tokens": 2048,
            "top_p": 0.9,
        },
    )


def _parse_json_response(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text output."""
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidates.extend(fenced)

    brace_match = re.search(r"(\{.*\})", text, flags=re.S)
    if brace_match:
        candidates.append(brace_match.group(1))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _generate_json(prompt: str, images: list[Image.Image]) -> dict[str, Any] | None:
    """Run a Gemini multimodal request and parse JSON output."""
    try:
        model = _init_gemini()
    except ImportError:
        warnings.warn(
            "[VisualTools] google-generativeai is not installed. "
            "Skipping Gemini-backed visual tools.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    contents: list[Any] = [prompt]
    contents.extend(images)

    try:
        response = model.generate_content(contents)
        if not getattr(response, "text", None):
            return None
        return _parse_json_response(response.text)
    except Exception:
        traceback.print_exc()
        return None


def understand_images(
    images: list[Image.Image],
    goal: str,
) -> dict[str, Any] | None:
    """Return a compact visual summary of the input images."""
    prompt = (
        "You are helping an image-editing agent inspect input images.\n"
        "Return JSON only with this schema:\n"
        "{"
        '"summaries": [{"image_index": 0, "summary": "...", "entities": ["..."]}],'
        '"overall_scene": "...",'
        '"risks": ["..."]'
        "}\n"
        f"User goal: {goal}"
    )
    return _generate_json(prompt, images)


def ocr_images(images: list[Image.Image]) -> dict[str, Any] | None:
    """Extract visible text from images."""
    prompt = (
        "Extract visible text from each image. Return JSON only with schema:\n"
        "{"
        '"texts": [{"image_index": 0, "text": "..."}],'
        '"notes": ["..."]'
        "}\n"
        "If no text is visible for an image, return an empty string for that image."
    )
    return _generate_json(prompt, images)


def score_edit_result(
    *,
    before_images: list[Image.Image],
    after_image: Image.Image,
    goal: str,
) -> dict[str, Any] | None:
    """Score the final edit result with Gemini."""
    prompt = (
        "You are a critic for an image-editing agent. Compare the source image(s) "
        "and the edited result against the user goal.\n"
        "Return JSON only with schema:\n"
        "{"
        '"score": 1,'
        '"verdict": "pass|soft_fail|hard_fail",'
        '"reasons": ["..."],'
        '"repair_advice": ["..."]'
        "}\n"
        "Use score 5 for excellent alignment and 1 for complete failure.\n"
        f"User goal: {goal}"
    )
    return _generate_json(prompt, before_images + [after_image])


__all__ = ["ocr_images", "score_edit_result", "understand_images"]
