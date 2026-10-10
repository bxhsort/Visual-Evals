from __future__ import annotations

import abc
import base64
import math
import os
import time
from io import BytesIO
from pathlib import Path

_DOTENV_LOADED = False


def load_provider_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    candidates = []
    if env_path := os.environ.get("PROVIDER_ENV_FILE"):
        candidates.append(Path(env_path))
    here = Path(__file__).resolve().parent.parent
    candidates.extend([here / ".env.providers", here / ".env"])

    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(candidate, override=False)
        except ImportError:
            _load_dotenv_fallback(candidate)
        return


def _load_dotenv_fallback(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_image_size_to_pixels(image_size: str) -> int:
    raw = str(image_size).strip().upper()
    presets = {"512": 512, "1K": 1024, "1.5K": 1536, "2K": 2048, "4K": 4096}
    if raw in presets:
        return presets[raw]
    if raw.endswith("K"):
        return int(float(raw[:-1]) * 1024)
    if raw.isdigit():
        return int(raw)
    raise ValueError(f"Unsupported image_size: {image_size!r}")


def image_size_to_resolution_token(image_size: str) -> str:
    """Map image_size to a '1K'/'2K'/'3K' style token; fallback to numeric."""
    pixels = parse_image_size_to_pixels(image_size)
    mapping = {1024: "1K", 2048: "2K", 3072: "3K"}
    return mapping.get(pixels, str(pixels))


def size_from_resolution(resolution: str, aspect_ratio: str | None, *, supported: dict[str, int]) -> str:
    """Compute a 'WxH' size string given a resolution tier and aspect ratio.

    supported maps tier -> max_pixel (e.g. {'1K': 1024**2}).
    """
    if not aspect_ratio:
        return resolution
    left, _, right = aspect_ratio.partition(":")
    ratio = int(left) / int(right)
    tier = resolution.upper()
    if tier not in supported:
        raise ValueError(f"Unsupported resolution {resolution!r}; supported: {list(supported)}")
    max_pixel = supported[tier]
    width = round(math.sqrt(max_pixel * ratio))
    height = round(math.sqrt(max_pixel / ratio))
    return f"{width}x{height}"


def decode_base64_image(payload: str):
    from PIL import Image

    if "," in payload and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload)
    return Image.open(BytesIO(raw)).convert("RGB")


def download_image(url: str, *, timeout: int = 120, max_retries: int = 3, retry_backoff: float = 2.0):
    from PIL import Image
    import requests

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"User-Agent": "generation_eval/1.0"})
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(min(retry_backoff * (2**attempt), 30.0))
    raise RuntimeError(f"Failed to download image from {url}: {last_exc}")


def post_json_with_retry(
    url: str,
    *,
    headers: dict,
    payload: dict,
    timeout: int,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> dict:
    import requests

    last_exc: Exception | None = None
    total_attempts = max(1, max_retries)
    for attempt in range(total_attempts):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if not resp.ok:
                try:
                    err_body = resp.json()
                except Exception:
                    err_body = resp.text[:2000]
                print(f"[WARN] {url} HTTP {resp.status_code}: {err_body}")
                resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            transient = status is None or status >= 500 or status in {408, 409, 429}
            if not transient or attempt >= total_attempts - 1:
                raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt >= total_attempts - 1:
                raise
        sleep_seconds = min(retry_backoff * (2**attempt), 30.0)
        print(f"[WARN] attempt {attempt + 1}/{total_attempts} failed: {last_exc}. Retrying in {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)
    raise RuntimeError(f"POST {url} failed after {total_attempts} attempts: {last_exc}")


class T2IProvider(abc.ABC):
    DEFAULT_MODEL: str = ""
    BASE_URL_ENV: str = ""
    DEFAULT_BASE_URL: str = ""
    API_KEY_ENV: str = ""

    def __init__(self, model_name: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL or "").rstrip("/")
        self.api_key = api_key or (os.environ.get(self.API_KEY_ENV, "") if self.API_KEY_ENV else "")
        if self.API_KEY_ENV and not self.api_key:
            raise RuntimeError(
                f"{self.__class__.__name__}: missing API key. Set env {self.API_KEY_ENV} "
                f"(via .env.providers or shell export)."
            )

    @abc.abstractmethod
    def generate_one(
        self,
        prompt: str,
        *,
        image_size: str,
        seed: int,
        aspect_ratio: str = "1:1",
        timeout: int = 600,
        negative_prompt: str = "",
        **extra,
    ):
        """Generate a single image and return a PIL.Image. Raise on failure."""
