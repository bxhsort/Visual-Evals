"""Load prompt-enhancement caches without exposing them to benchmark judges."""

import hashlib
import json


REQUIRED_CAPTION_KEYS = {
    "subjects",
    "subject_details",
    "background_setting",
    "lighting",
    "text_and_signage_elements",
    "resolution",
    "aspect_ratio",
    "comprehensive_t2i_caption",
}


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_pe_prompts(path):
    prompts = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") is None:
                raise ValueError(f"Enhanced prompt has no id in {path}:{line_number}")
            prompt_id = str(row["id"])
            if prompt_id in prompts:
                raise ValueError(
                    f"Duplicate enhanced prompt id {prompt_id!r} in {path}:{line_number}"
                )
            caption = row.get("caption_json")
            if not isinstance(caption, dict):
                raise ValueError(
                    f"Enhanced prompt {prompt_id!r} has no caption_json object"
                )
            missing = sorted(REQUIRED_CAPTION_KEYS - set(caption))
            if missing:
                raise ValueError(f"Enhanced prompt {prompt_id!r} is missing keys: {missing}")
            if caption["resolution"] != {"H": 2048, "W": 2048}:
                raise ValueError(f"Enhanced prompt {prompt_id!r} is not 2048x2048")
            if caption["aspect_ratio"] != "1,1":
                raise ValueError(
                    f"Enhanced prompt {prompt_id!r} does not use aspect ratio 1,1"
                )
            comprehensive_caption = caption["comprehensive_t2i_caption"]
            if not isinstance(comprehensive_caption, str) or not comprehensive_caption.strip():
                raise ValueError(
                    f"Enhanced prompt {prompt_id!r} has an empty comprehensive caption"
                )
            if not isinstance(row.get("source_prompt_sha256"), str):
                raise ValueError(f"Enhanced prompt {prompt_id!r} has no source hash")
            prompts[prompt_id] = row
    if not prompts:
        raise ValueError(f"Enhanced prompt cache is empty: {path}")
    return prompts


def resolve_pe_prompt(original_prompt, prompt_id, prompts):
    """Return serialized structured JSON for one source prompt, or the raw prompt."""
    if prompts is None:
        return original_prompt
    key = str(prompt_id)
    if key not in prompts:
        raise KeyError(f"Enhanced prompt id {key!r} is missing")
    row = prompts[key]
    if row.get("prompt") != original_prompt:
        raise ValueError(f"Enhanced prompt source text changed for id {key!r}")
    expected_hash = row["source_prompt_sha256"]
    if expected_hash != _sha256(original_prompt):
        raise ValueError(f"Enhanced prompt source hash changed for id {key!r}")
    return json.dumps(row["caption_json"], ensure_ascii=False)
