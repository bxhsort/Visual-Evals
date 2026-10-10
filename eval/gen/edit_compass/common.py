import os
from pathlib import Path


PART_TASKS = {
    "Part1": [
        "ADD",
        "Remove",
        "Replace",
        "Change_color_size",
        "Style_Transfer",
        "Extract",
        "Change_Material",
        "Change_Background",
        "Visual_Text_CN",
        "Visual_Text_EN",
    ],
    "Part2": [
        "Action",
        "Move",
        "Swap",
        "Object_Interaction",
        "Emotion_Change",
    ],
}


METRIC_FIELD = {
    "IC": "Identity_Consistency",
    "IF": "Instrcution_Following",
    "VQ": "Visual_Quality",
    "URC": "Unedited_Region_Consistency",
    "WA": "World_Awareness",
}


def parse_parts(parts):
    if isinstance(parts, str):
        parts = parts.replace(",", " ").split()
    parsed = []
    for part in parts:
        if part not in PART_TASKS:
            raise ValueError(f"Unsupported Edit-Compass part: {part}; expected one of {sorted(PART_TASKS)}")
        parsed.append(part)
    return parsed


def iter_task_configs(data_root, save_root, parts):
    data_root = Path(data_root)
    save_root = Path(save_root)
    for part in parse_parts(parts):
        for task_name in PART_TASKS[part]:
            yield {
                "part": part,
                "task_name": task_name,
                "json_path": data_root / part / task_name / f"{task_name}.json",
                "save_dir": save_root / part / task_name,
            }


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
