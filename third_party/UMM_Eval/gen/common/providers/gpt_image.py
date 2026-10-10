from __future__ import annotations

from . import register_provider
from .base import (
    T2IProvider,
    decode_base64_image,
    image_size_to_resolution_token,
    post_json_with_retry,
    size_from_resolution,
)


@register_provider("gpt_image")
class GPTImageProvider(T2IProvider):
    DEFAULT_MODEL = "gpt-image-2"
    BASE_URL_ENV = "GPT_IMAGE_BASE_URL"
    DEFAULT_BASE_URL = "http://15.204.107.114:3009/v1"
    API_KEY_ENV = "PPAPI_GPT_API_KEY"

    SUPPORTED_RESOLUTIONS = {"1K": 1024**2, "2K": 2048**2}

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
        if resolution in self.SUPPORTED_RESOLUTIONS:
            size = size_from_resolution(resolution, aspect_ratio, supported=self.SUPPORTED_RESOLUTIONS)
        else:
            size = "auto"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": 1,
            "size": size,
            "response_format": "b64_json",
        }
        url = f"{self.base_url}/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = post_json_with_retry(
            url, headers=headers, payload=payload, timeout=timeout, max_retries=max_retries
        )

        items = data.get("data") or []
        if not items:
            raise RuntimeError(f"gpt-image response missing 'data': {data}")
        b64 = items[0].get("b64_json")
        if not b64:
            raise RuntimeError(f"gpt-image response missing b64_json: {items[0]}")
        return decode_base64_image(b64)
