from __future__ import annotations

import os

from .base import T2IProvider, load_provider_dotenv


_PROVIDER_REGISTRY: dict[str, type[T2IProvider]] = {}


def register_provider(name: str):
    def decorator(cls: type[T2IProvider]) -> type[T2IProvider]:
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return decorator


def _import_builtins() -> None:
    from . import gpt_image  # noqa: F401
    from . import lightllm  # noqa: F401
    from . import nano_banana  # noqa: F401
    from . import qwen_image  # noqa: F401
    from . import seedream  # noqa: F401


def _route_by_model_name(model_name: str) -> str:
    name = model_name.lower()
    if name.startswith("gpt-image"):
        return "gpt_image"
    if name.startswith("nano-banana"):
        return "nano_banana"
    if name.startswith("doubao-seedream") or name.startswith("seedream"):
        return "seedream"
    if name.startswith("qwen-image"):
        return "qwen_image"
    return "lightllm"


def get_provider(model_name: str, provider_override: str | None = None, **kwargs) -> T2IProvider:
    load_provider_dotenv()
    _import_builtins()

    provider_key = provider_override or os.environ.get("PROVIDER") or _route_by_model_name(model_name)
    provider_key = provider_key.strip().lower().replace("-", "_")

    if provider_key not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider {provider_key!r}. Registered: {sorted(_PROVIDER_REGISTRY)}"
        )
    return _PROVIDER_REGISTRY[provider_key](model_name=model_name, **kwargs)


__all__ = ["T2IProvider", "get_provider", "register_provider"]
