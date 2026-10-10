from __future__ import annotations

import os

from . import register_provider
from .base import T2IProvider


@register_provider("lightllm")
class LightLLMProvider(T2IProvider):
    DEFAULT_MODEL = ""
    BASE_URL_ENV = "LIGHTLLM_BASE_URL"
    DEFAULT_BASE_URL = "http://localhost:8000/v1"
    API_KEY_ENV = ""

    def __init__(self, model_name: str | None = None, base_url: str | None = None, api_key: str | None = None):
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key)
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "dummy")

    def generate_one(
        self,
        prompt: str,
        *,
        image_size: str,
        seed: int,
        aspect_ratio: str = "1:1",
        timeout: int = 600,
        negative_prompt: str = "",
        guidance_scale: float | None = None,
        cfg_norm: str | None = "global",
        timestep_shift: float | None = None,
        image_type: str = "png",
        enable_thinking: bool = False,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        return_message: bool = False,
        **_extra,
    ):
        from inference_api_common import (
            build_t2i_image_config,
            chat_t2i,
            decode_image_from_message,
        )

        image_config = build_t2i_image_config(
            image_size,
            image_type,
            seed,
            aspect_ratio=aspect_ratio,
            guidance_scale=guidance_scale,
            cfg_norm=cfg_norm,
            timestep_shift=timestep_shift,
        )
        message = chat_t2i(
            base_url=self.base_url,
            api_key=self.api_key,
            model_name=self.model_name,
            user_text=prompt,
            image_config=image_config,
            enable_thinking=enable_thinking,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        image = decode_image_from_message(message)
        if image is None:
            raise RuntimeError("LightLLM returned no image")
        if return_message:
            return image, message
        return image
