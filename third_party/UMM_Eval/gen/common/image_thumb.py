"""Downscale + base64-encode an image to a ``data:`` URL.

Shared by ``qualitative_review`` (vision-call payload — keeps requests under the tokenhub
gateway's 50MB limit even at 12 full-res images) and ``build_report`` (inline HTML
thumbnails). Falls back to raw bytes if PIL or the file is unavailable — never raises.
"""
from __future__ import annotations

import base64
import io
import os

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp", ".gif": "image/gif"}


def to_data_url(path, max_edge=768, fmt="JPEG", quality=85):
    """Return a ``data:`` URL for ``path``, downscaled so its long edge <= ``max_edge``.

    Re-encodes as JPEG (much smaller than full-res PNG) for the payload. On any failure
    (PIL missing, unreadable file) falls back to the raw bytes; if even that fails returns
    an empty string. Never raises.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest > max_edge:
                scale = max_edge / float(longest)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
            buf = io.BytesIO()
            im.save(buf, format=fmt, quality=quality)
            data = buf.getvalue()
        mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else "image/png"
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    except Exception:
        try:
            ext = os.path.splitext(path)[1].lower()
            mime = _MIME.get(ext, "image/png")
            with open(path, "rb") as f:
                return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")
        except Exception:
            return ""
