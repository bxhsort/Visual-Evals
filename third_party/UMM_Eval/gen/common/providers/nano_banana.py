from __future__ import annotations

from . import register_provider
from .base import (
    T2IProvider,
    decode_base64_image,
    image_size_to_resolution_token,
    post_json_with_retry,
)


@register_provider("nano_banana")
class NanoBananaProvider(T2IProvider):
    DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
    BASE_URL_ENV = "PPAPI_NANO_BASE_URL"
    DEFAULT_BASE_URL = "http://15.204.107.114:3009"
    API_KEY_ENV = "PPAPI_API_KEY_NANO"

    def generate_one(
        self,
        prompt: str,
        *,
        image_size: str,
        seed: int,
        aspect_ratio: str = "1:1",
        timeout: int = 600,
        negative_prompt: str = "",
        max_retries: int = 3,
        **_extra,
    ):
        resolution = image_size_to_resolution_token(image_size)
        payload = {
            "contents": [{"role": "USER", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_modalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "imageSize": resolution.upper(),
                    "aspectRatio": aspect_ratio,
                },
            },
        }
        url = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        data = post_json_with_retry(
            url, headers=headers, payload=payload, timeout=timeout, max_retries=max_retries
        )

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"nano-banana response missing 'candidates': {data}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return decode_base64_image(inline["data"])
        raise RuntimeError(f"nano-banana response has no image data: {data}")
