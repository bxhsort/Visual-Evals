from __future__ import annotations

from . import register_provider
from .base import (
    T2IProvider,
    download_image,
    image_size_to_resolution_token,
    parse_image_size_to_pixels,
    post_json_with_retry,
    size_from_resolution,
)


@register_provider("seedream")
class SeedreamProvider(T2IProvider):
    DEFAULT_MODEL = "doubao-seedream-4-5-251128"
    BASE_URL_ENV = "ARK_BASE_URL"
    DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
    API_KEY_ENV = "ARK_API_KEY"

    SUPPORTED_RESOLUTIONS = {"1K": 1024**2, "2K": 2048**2, "3K": 3072**2}
    # Per-model minimum total pixels enforced server-side by ARK.
    # seedream-4-5 / 5-0 reject sizes below 3,686,400 px (~1920x1920) → bump 1K to 2K.
    MODEL_MIN_PIXELS = {"4-5": 3_686_400, "5-0": 3_686_400}

    def _enforce_min_resolution(self, resolution: str) -> str:
        for fragment, min_pixels in self.MODEL_MIN_PIXELS.items():
            if fragment not in self.model_name:
                continue
            current_pixels = self.SUPPORTED_RESOLUTIONS.get(resolution)
            if current_pixels is None or current_pixels >= min_pixels:
                return resolution
            for tier in ("2K", "3K"):
                if self.SUPPORTED_RESOLUTIONS.get(tier, 0) >= min_pixels:
                    print(
                        f"[WARN] {self.model_name} requires >= {min_pixels} px; "
                        f"bumping resolution {resolution} -> {tier}"
                    )
                    return tier
        return resolution

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
        watermark: bool = False,
        **_extra,
    ):
        resolution = image_size_to_resolution_token(image_size)
        resolution = self._enforce_min_resolution(resolution)
        if resolution in self.SUPPORTED_RESOLUTIONS:
            size = size_from_resolution(resolution, aspect_ratio, supported=self.SUPPORTED_RESOLUTIONS)
        else:
            size = resolution

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "size": size,
            "response_format": "url",
            "watermark": watermark,
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
            raise RuntimeError(f"seedream response missing 'data': {data}")
        image_url = items[0].get("url")
        if not image_url:
            raise RuntimeError(f"seedream response missing url: {items[0]}")
        image = download_image(image_url, timeout=120, max_retries=max_retries)

        target_pixels = parse_image_size_to_pixels(image_size)
        if image.size != (target_pixels, target_pixels):
            from PIL import Image as _Image
            image = image.resize((target_pixels, target_pixels), _Image.LANCZOS)
        return image
