"""Offline tests for Qwen-Image-Bench TokenHub response handling."""

import base64
import importlib.util
import io
import sys
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "gen" / "common"))
MODULE_PATH = Path(__file__).with_name("sota_tokenhub_gen.py")
SPEC = importlib.util.spec_from_file_location("qib_sota_tokenhub_gen", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _png_base64() -> str:
    output = io.BytesIO()
    Image.new("RGBA", (3, 2), (10, 20, 30, 128)).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    "encoded",
    [
        _png_base64(),
        f"data:image/png;base64,{_png_base64()}",
        f"DATA:IMAGE/PNG;BASE64,{_png_base64()}",
    ],
)
def test_decode_image_accepts_raw_base64_and_image_data_urls(encoded):
    image = module._decode_image({"b64_json": encoded})

    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_decode_image_accepts_line_wrapped_raw_base64():
    encoded = _png_base64()
    wrapped = "\n".join(encoded[index:index + 20] for index in range(0, len(encoded), 20))

    assert module._decode_image({"b64_json": wrapped}).size == (3, 2)


@pytest.mark.parametrize(
    "encoded",
    [
        "data:text/plain;base64,SGVsbG8=",
        "data:image/png,not-base64",
        "data:image/png;base64,not-valid-base64!",
        base64.b64encode(b"not an image").decode("ascii"),
    ],
)
def test_decode_image_rejects_invalid_image_payloads(encoded):
    with pytest.raises((ValueError, OSError)):
        module._decode_image({"b64_json": encoded})


def test_main_returns_nonzero_when_generation_fails(monkeypatch, tmp_path):
    class ImmediateFuture:
        def result(self):
            return "sample-1", "failed", "invalid image response"

    class InlineExecutor:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def submit(self, *_args):
            return ImmediateFuture()

    monkeypatch.setattr(module, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(module, "as_completed", lambda futures: futures)
    monkeypatch.setattr(module, "load_api_key", lambda: "unused-test-key")
    monkeypatch.setattr(
        module,
        "read_jsonl",
        lambda _path: [{"ID": "sample-1", "prompt": "offline test"}],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(MODULE_PATH), "--mode", "en", "--output_base", str(tmp_path), "--workers", "1"],
    )

    assert module.main() == 1
