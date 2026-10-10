from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
from io import BytesIO


# Fallback output root only; launchers always pass an explicit --output_path, so this
# is rarely used. Override via UMM_EVAL_OUTPUT_ROOT; defaults to CWD (portable, no fixed dir).
DEFAULT_WORK_ROOT = os.environ.get("UMM_EVAL_OUTPUT_ROOT", os.getcwd())
DEFAULT_OUTPUT_PARENT = ""
IMAGE_SIZE_PRESETS = {
    "512": 512,
    "1K": 1024,
    "1.5K": 1536,
    "2K": 2048,
    "4K": 4096,
}

SYSTEM = (
    "You are an image generation and editing assistant that accurately understands and executes user intent.\n\n"
    "You support two modes:\n\n"
    "1. Think Mode:\n"
    "If the task requires reasoning, you MUST start with a <think></think> block. Put all reasoning inside the block using plain text. "
    "DO NOT include any image tags. Keep it reasonable and directly useful for producing the final image.\n\n"
    "2. Non-Think Mode:\n"
    "If no reasoning is needed, directly produce the final image.\n\n"
    "Task Types:\n\n"
    "A. Text-to-Image Generation:\n"
    "- Generate a high-quality image based on the user's description.\n"
    "- Ensure visual clarity, semantic consistency, and completeness.\n"
    "- DO NOT introduce elements that contradict or override the user's intent.\n\n"
    "B. Image Editing:\n"
    "- Use the provided image(s) as input or reference for modification or transformation.\n"
    "- The result can be an edited image or a new image based on the reference(s).\n"
    "- Preserve all unspecified attributes unless explicitly changed.\n\n"
    "General Rules:\n"
    "- For any visible text in the image, follow the language specified for the rendered text in the user's description, not the language of the prompt. "
    "If no language is specified, use the user's input language."
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[\\/:\s]+", "_", value.strip())
    return sanitized or "api"


def format_numeric(value):
    return f"{value:g}"


def normalize_image_size_spec(value):
    raw_value = str(value).strip()
    compact_value = raw_value.replace(" ", "").upper()

    if compact_value in IMAGE_SIZE_PRESETS:
        return compact_value

    if compact_value.endswith("K") and is_positive_number(compact_value[:-1]):
        return canonicalize_numeric_image_size(int(float(compact_value[:-1]) * 1024))

    normalized_value = compact_value.replace("*", "X")
    if "X" in normalized_value:
        width_str, height_str = normalized_value.split("X", 1)
        if not width_str.isdigit() or not height_str.isdigit():
            raise argparse.ArgumentTypeError(
                f"image_size must be a preset like 1K/1.5K/2K or a square spec like 1024x1024, got {value!r}"
            )
        width = int(width_str)
        height = int(height_str)
        if width <= 0 or height <= 0:
            raise argparse.ArgumentTypeError(
                f"image_size must be positive, got {value!r}"
            )
        if width != height:
            raise argparse.ArgumentTypeError(
                f"image_size must be square, got {value!r}"
            )
        return canonicalize_numeric_image_size(width)

    if compact_value.isdigit():
        width = int(compact_value)
        if width <= 0:
            raise argparse.ArgumentTypeError(
                f"image_size must be positive, got {value!r}"
            )
        return canonicalize_numeric_image_size(width)

    raise argparse.ArgumentTypeError(
        f"image_size must be a preset like 1K/1.5K/2K or a square spec like 1024x1024, got {value!r}"
    )


def canonicalize_numeric_image_size(width: int) -> str:
    if width == 1024:
        return "1K"
    if width == 1536:
        return "1.5K"
    if width == 2048:
        return "2K"
    if width == 4096:
        return "4K"
    return str(width)


def image_size_spec_to_square_pixels(image_size_spec: str) -> int:
    compact_value = str(image_size_spec).strip().upper()
    if compact_value in IMAGE_SIZE_PRESETS:
        return IMAGE_SIZE_PRESETS[compact_value]
    if compact_value.endswith("K") and is_positive_number(compact_value[:-1]):
        return int(float(compact_value[:-1]) * 1024)
    if compact_value.isdigit():
        return int(compact_value)
    raise ValueError(f"Unsupported image_size spec: {image_size_spec!r}")


def is_positive_number(value: str) -> bool:
    try:
        return float(value) > 0
    except ValueError:
        return False


def parse_target_ints(raw_value):
    if raw_value is None:
        return None

    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    return set(values) if values else None


def parse_target_strings(raw_value):
    if raw_value is None:
        return None

    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if part:
            values.append(part)
    return set(values) if values else None


def parse_csv(raw_value, cast_fn=str):
    if raw_value is None:
        return None

    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if part:
            values.append(cast_fn(part))
    return values or None


def parse_base_urls(raw_value):
    urls = parse_csv(raw_value, str)
    if urls is None:
        return None

    normalized_urls = []
    for url in urls:
        normalized_url = str(url).strip()
        if normalized_url:
            normalized_urls.append(normalized_url)
    return normalized_urls or None


def select_base_url(base_url, base_urls=None, dispatch_index=0):
    if base_urls:
        return base_urls[int(dispatch_index) % len(base_urls)]
    return base_url


def str_to_bool(value):
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_output_root(image_size_pixels, cfg_scale, timestep_shift):
    if math.isclose(cfg_scale, 7.0):
        output_root = f"eval_output_{image_size_pixels}"
    else:
        output_root = f"eval_output_cfg{format_numeric(cfg_scale)}_{image_size_pixels}"

    if not math.isclose(timestep_shift, 1.0):
        output_root = f"{output_root}_ts{format_numeric(timestep_shift)}"
    if timestep_shift < 1.0:
        output_root = f"{output_root}_noshift"

    return output_root


def build_edit_output_folder(
    txt_scale, img_cfg, cfg_norm, timestep_shift, image_size=None
):
    if image_size is None or image_size == 512:
        output_folder = f"eval_output_edit_txtcfg{txt_scale:g}_imgcfg{img_cfg:g}"
    else:
        output_folder = (
            f"eval_output_edit_{image_size}_txtcfg{txt_scale:g}_imgcfg{img_cfg:g}"
        )

    if not math.isclose(timestep_shift, 1.0):
        output_folder = f"{output_folder}_ts{format_numeric(timestep_shift)}"
    if timestep_shift < 1.0:
        output_folder = f"{output_folder}_noshift"
    if cfg_norm != "none":
        output_folder = f"{output_folder}"

    return output_folder


def build_rise_output_folder(txt_scale, img_cfg, cfg_norm, timestep_shift):
    output_folder = f"eval_output_edit_txtcfg{txt_scale:g}_imgcfg{img_cfg:g}"
    if not math.isclose(timestep_shift, 1.0):
        output_folder = f"{output_folder}_ts{format_numeric(timestep_shift)}"
    if timestep_shift < 1.0:
        output_folder = f"{output_folder}_noshift"
    if cfg_norm != "none":
        output_folder = f"{output_folder}_{cfg_norm}"
    return output_folder


def resolve_output_dir(
    output_path,
    output_root,
    model_name,
    output_subdir,
    work_root=DEFAULT_WORK_ROOT,
    output_parent=DEFAULT_OUTPUT_PARENT,
):
    if output_path is not None:
        return output_path
    return os.path.join(
        work_root,
        output_root,
        output_parent,
        sanitize_path_component(model_name),
        output_subdir,
    )


def default_num_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def get_progress_bar():
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:

        def progress_bar(iterable, **kwargs):
            return iterable

        return progress_bar


def get_runtime_shard():
    shard_count = int(os.environ.get("API_SHARD_COUNT", "1"))
    shard_rank = int(os.environ.get("API_SHARD_RANK", "0"))
    if shard_count <= 0:
        raise ValueError(f"API_SHARD_COUNT must be positive, got {shard_count}")
    if not 0 <= shard_rank < shard_count:
        raise ValueError(
            f"API_SHARD_RANK must be in [0, API_SHARD_COUNT), got shard_rank={shard_rank}, shard_count={shard_count}"
        )
    return shard_count, shard_rank


def shard_data(data):
    shard_count, shard_rank = get_runtime_shard()
    if shard_count == 1:
        return data, shard_count, shard_rank
    return data[shard_rank::shard_count], shard_count, shard_rank


def pil_image_to_data_url(image, image_type: str = "png") -> str:
    subtype = image_type.lower()
    pil_format = "JPEG" if subtype in {"jpg", "jpeg"} else subtype.upper()
    mime_subtype = "jpeg" if subtype == "jpg" else subtype

    image_to_save = image.convert("RGB") if mime_subtype == "jpeg" else image
    buffer = BytesIO()
    image_to_save.save(buffer, format=pil_format)
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{mime_subtype};base64,{base64_image}"


def decode_image_from_message(message: dict):
    from PIL import Image

    images = message.get("images") or []
    for item in images:
        if not isinstance(item, dict):
            continue
        url = (item.get("image_url") or {}).get("url")
        if not url:
            continue
        match = re.match(r"data:image/[\w+.-]+;base64,(.+)", url, re.DOTALL)
        if not match:
            continue
        raw = base64.b64decode(match.group(1))
        return Image.open(BytesIO(raw)).convert("RGB")
    return None


def make_grid(images, grid_rows: int, grid_cols: int, pad: int = 2):
    from PIL import Image

    width, height = images[0].size
    grid = Image.new(
        "RGB",
        (
            grid_cols * width + (grid_cols - 1) * pad,
            grid_rows * height + (grid_rows - 1) * pad,
        ),
    )
    for idx, image in enumerate(images):
        row, col = divmod(idx, grid_cols)
        y = row * (height + pad)
        x = col * (width + pad)
        grid.paste(image, (x, y))
    return grid


def make_side_by_side_visualization(input_image, output_image):
    from PIL import Image

    vis_image = Image.new(
        "RGB",
        (
            input_image.width + output_image.width,
            max(input_image.height, output_image.height),
        ),
        (255, 255, 255),
    )
    vis_image.paste(input_image, (0, 0))
    vis_image.paste(output_image, (input_image.width, 0))
    return vis_image


def round_by_factor(number, factor):
    return round(number / factor) * factor


def ceil_by_factor(number, factor):
    return math.ceil(number / factor) * factor


def floor_by_factor(number, factor):
    return math.floor(number / factor) * factor


def smart_resize(
    height, width, factor=32, min_pixels=256 * 32 * 32, max_pixels=16384 * 32 * 32
):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def simplify_aspect_ratio(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def apply_image_sampling_args(
    image_config: dict,
    guidance_scale: float | None = None,
    cfg_norm: str | None = "global",
    timestep_shift: float | None = None,
) -> dict:
    config = dict(image_config)
    if guidance_scale is not None:
        config["guidance_scale"] = guidance_scale
    if cfg_norm is not None:
        config["cfg_norm"] = cfg_norm
    if timestep_shift is not None:
        config["timestep_shift"] = timestep_shift
    return config


def build_t2i_image_config(
    image_size: str,
    image_type: str,
    seed: int,
    aspect_ratio: str = "1:1",
    guidance_scale: float | None = None,
    cfg_norm: str | None = "global",
    timestep_shift: float | None = None,
) -> dict:
    image_size_pixels = image_size_spec_to_square_pixels(image_size)
    return apply_image_sampling_args(
        {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "image_type": image_type,
            "seed": seed,
            "height": image_size_pixels,
            "width": image_size_pixels,
            "dynamic_resolution": False,
        },
        guidance_scale=guidance_scale,
        cfg_norm=cfg_norm,
        timestep_shift=timestep_shift,
    )


def build_image_config(
    width: int,
    height: int,
    image_size: str,
    image_type: str,
    seed: int,
    guidance_scale: float | None = None,
    cfg_norm: str | None = "global",
    timestep_shift: float | None = None,
    dynamic_resolution: bool = True,
    output_width: int = -1,
    output_height: int = -1,
) -> dict:
    return apply_image_sampling_args(
        {
            "aspect_ratio": simplify_aspect_ratio(width, height),
            "image_size": image_size,
            "image_type": image_type,
            "seed": seed,
            "height": output_height,
            "width": output_width,
            "dynamic_resolution": dynamic_resolution,
        },
        guidance_scale=guidance_scale,
        cfg_norm=cfg_norm,
        timestep_shift=timestep_shift,
    )


def normalize_text_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "reasoning", "content"):
            if key in content:
                return normalize_text_content(content[key])
        return str(content)
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
            else:
                texts.append(normalize_text_content(item))
        return "\n".join(text for text in texts if text).strip()
    return str(content)


def cleanup_assistant_text(text: str) -> str:
    cleaned = re.sub(r"</?image>", "", text or "", flags=re.IGNORECASE)
    return cleaned.strip()


def extract_tagged_think_content(text: str) -> str:
    matches = re.findall(
        r"<think>(.*?)</think>", text or "", flags=re.DOTALL | re.IGNORECASE
    )
    normalized_matches = [match.strip() for match in matches if match and match.strip()]
    return "\n\n".join(normalized_matches)


def extract_think_payload(message: dict) -> dict:
    reasoning_content = cleanup_assistant_text(
        normalize_text_content(
            message.get("reasoning_content")
            or message.get("thinking")
            or message.get("reasoning")
        )
    )
    assistant_content = cleanup_assistant_text(
        normalize_text_content(message.get("content"))
    )
    tagged_think_content = extract_tagged_think_content(assistant_content)
    think_content = reasoning_content or tagged_think_content or assistant_content
    return {
        "assistant_content": assistant_content,
        "think_content": think_content.strip(),
    }


def build_think_trace(think_payloads):
    entries = []
    merged_think_entries = []
    merged_assistant_entries = []

    for request_index, payload in enumerate(think_payloads):
        payload = payload or {}
        think_content = (payload.get("think_content") or "").strip()
        assistant_content = (payload.get("assistant_content") or "").strip()
        entries.append(
            {
                "micro_batch_index": request_index,
                "think_content": think_content,
                "assistant_content": assistant_content,
            }
        )
        if think_content:
            merged_think_entries.append(think_content)
        if assistant_content:
            merged_assistant_entries.append(assistant_content)

    merged_think_content = ""
    if merged_think_entries:
        unique_entries = []
        for text in merged_think_entries:
            if text not in unique_entries:
                unique_entries.append(text)
        if len(unique_entries) == 1:
            merged_think_content = unique_entries[0]
        else:
            merged_think_content = "\n\n".join(
                f"[micro_batch {entry['micro_batch_index']}]\n{entry['think_content']}"
                for entry in entries
                if entry["think_content"]
            )

    merged_assistant_content = ""
    if merged_assistant_entries:
        unique_entries = []
        for text in merged_assistant_entries:
            if text not in unique_entries:
                unique_entries.append(text)
        if len(unique_entries) == 1:
            merged_assistant_content = unique_entries[0]
        else:
            merged_assistant_content = "\n\n".join(
                f"[micro_batch {entry['micro_batch_index']}]\n{entry['assistant_content']}"
                for entry in entries
                if entry["assistant_content"]
            )

    return {
        "think_content": merged_think_content,
        "assistant_content": merged_assistant_content,
        "micro_batch_think_contents": entries,
    }


def chat_t2i(
    base_url: str,
    api_key: str,
    model_name: str,
    user_text: str,
    image_config: dict,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    timeout: int = 600,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_text},
        ],
        "modalities": ["image"],
        "stream": False,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "image_config": image_config,
    }
    return _post_chat_completion_with_retry(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def chat_it2i(
    base_url: str,
    api_key: str,
    model_name: str,
    user_text: str,
    input_image,
    image_config: dict,
    input_image_type: str = "png",
    enable_thinking: bool = False,
    temperature: float = 0.0,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    timeout: int = 600,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict:
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": pil_image_to_data_url(input_image, input_image_type)
                        },
                    },
                ],
            },
        ],
        "modalities": ["image"],
        "stream": False,
        "n": 1,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "image_config": image_config,
    }
    return _post_chat_completion_with_retry(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )


def _post_chat_completion_with_retry(
    base_url: str,
    api_key: str,
    payload: dict,
    timeout: int,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> dict:
    import requests

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    total_attempts = max(1, int(max_retries) + 1)
    last_exc = None

    for attempt_index in range(total_attempts):
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                raise ValueError("No choices returned by API")
            message = choices[0].get("message")
            if message is None:
                raise ValueError("No message returned by API")
            return message
        except requests.HTTPError as exc:
            last_exc = exc
            status_code = exc.response.status_code if exc.response is not None else None
            should_retry = (
                status_code is None
                or status_code >= 500
                or status_code in {408, 409, 429}
            )
            if not should_retry or attempt_index >= total_attempts - 1:
                raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt_index >= total_attempts - 1:
                raise

        sleep_seconds = min(retry_backoff_seconds * (2**attempt_index), 30.0)
        print(
            f"[WARN] API request attempt {attempt_index + 1}/{total_attempts} failed: {last_exc}. "
            f"Retrying in {sleep_seconds:.1f}s"
        )
        time.sleep(sleep_seconds)

    raise RuntimeError(
        f"API request failed after {total_attempts} attempts: {last_exc}"
    )
