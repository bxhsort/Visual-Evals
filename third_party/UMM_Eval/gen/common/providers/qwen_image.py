from __future__ import annotations

import os
from typing import Any

from . import register_provider
from .base import (
    T2IProvider,
    download_image,
    parse_image_size_to_pixels,
)


def _collect_image_urls(obj: Any) -> list[str]:
    urls: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (
                key in {"url", "image", "image_url"}
                and isinstance(value, str)
                and value.startswith(("http://", "https://"))
            ):
                urls.append(value)
            else:
                urls.extend(_collect_image_urls(value))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(_collect_image_urls(item))
    return urls


@register_provider("qwen_image")
class QwenImageProvider(T2IProvider):
    DEFAULT_MODEL = "qwen-image-2.0-pro"
    BASE_URL_ENV = "DASHSCOPE_BASE_URL"
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
    API_KEY_ENV = "DASHSCOPE_API_KEY"

    def __init__(self, model_name: str | None = None, base_url: str | None = None, api_key: str | None = None):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key)
        try:
            import dashscope  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install dependency first: pip install dashscope") from exc
        import dashscope as _dashscope
        _dashscope.base_http_api_url = self.base_url

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
        prompt_extend: bool = False,
        watermark: bool = False,
        **_extra,
    ):
        from dashscope import MultiModalConversation
        import time

        pixels = parse_image_size_to_pixels(image_size)
        size = f"{pixels}*{pixels}"

        kwargs = dict(
            api_key=self.api_key,
            model=self.model_name,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            result_format="message",
            stream=False,
            watermark=watermark,
            prompt_extend=prompt_extend,
            size=size,
            seed=seed,
        )
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        last_exc: Exception | None = None
        for attempt in range(max(1, max_retries)):
            try:
                response = MultiModalConversation.call(**kwargs)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"DashScope error: HTTP {response.status_code}, "
                        f"code={getattr(response, 'code', None)}, "
                        f"message={getattr(response, 'message', None)}"
                    )
                urls = _collect_image_urls(response.output)
                if not urls:
                    raise RuntimeError(f"DashScope response has no image URL: {response.output}")
                return download_image(urls[0], timeout=120, max_retries=max_retries)
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries - 1:
                    raise
                sleep_seconds = min(2.0 * (2**attempt), 30.0)
                print(f"[WARN] qwen-image attempt {attempt + 1}/{max_retries} failed: {exc}. Retrying in {sleep_seconds:.1f}s")
                time.sleep(sleep_seconds)
        raise RuntimeError(f"qwen-image failed after {max_retries} attempts: {last_exc}")
