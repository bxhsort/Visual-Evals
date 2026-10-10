import os
import re
import pathlib

TOKENHUB_BASE_URL = os.environ.get(
    "EVAL_API_BASE_URL",
    os.environ.get("OPENAI_BASE_URL", "https://tokenhub.sensetime.com/v1"),
)
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _read_token_hub_key(path):
    """Read ``token_hub.api_key`` from a small YAML file without logging it."""
    raw = pathlib.Path(path).read_text()
    section = re.search(
        r"(?ms)^token_hub\s*:\s*$\n(?P<body>(?:^[ \t]+.*(?:\n|$))*)",
        raw,
    )
    if not section:
        raise RuntimeError(f"token_hub section not found in {path}")
    match = re.search(r"(?m)^[ \t]+api_key\s*:\s*(sk-[A-Za-z0-9]+)\s*$", section.group("body"))
    if not match:
        raise RuntimeError(f"token_hub.api_key not found in {path}")
    return match.group(1)


def resolve_eval_api_key(explicit=None):
    """Resolve the eval judge API key.

    Precedence: explicit arg > env API_KEY > env EVAL_API_KEY_FILE's
    token_hub.api_key > api_key.yaml:token_hub_eval.
    The key is read at runtime and never printed or logged.
    """
    if explicit:
        return explicit
    env = os.environ.get("API_KEY")
    if env:
        return env
    key_file = os.environ.get("EVAL_API_KEY_FILE")
    if key_file:
        return _read_token_hub_key(key_file)
    default_key_file = _REPO_ROOT / "api_key.yaml"
    if not default_key_file.exists():
        raise RuntimeError("API key is required; set API_KEY or EVAL_API_KEY_FILE")
    raw = default_key_file.read_text()
    m = re.search(r"token_hub_eval\s*:?\s*(sk-[A-Za-z0-9]+)", raw)
    if not m:
        raise RuntimeError("token_hub_eval key not found in api_key.yaml")
    return m.group(1)
