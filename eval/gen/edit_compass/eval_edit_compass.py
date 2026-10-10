import argparse
import base64
import json
import os
import re
import sys
import threading
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from eval.gen.edit_compass.common import METRIC_FIELD, iter_task_configs


DG2_ROOT = Path(__file__).resolve().parents[3]


class OpenAICompatibleEvalWrapper:
    def __init__(self, model_name, api_key, api_base=None, temperature=0.0, max_image_side=2048):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=api_base or None)
        self.model_name = model_name
        self.temperature = temperature
        self.max_image_side = max_image_side

    def encode_image(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            max_side = max(width, height)
            if self.max_image_side > 0 and max_side > self.max_image_side:
                scale = self.max_image_side / max_side
                new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def create_message(self, text, image_urls):
        text_splits = text.split("<image>")
        if len(text_splits) != len(image_urls) + 1:
            raise ValueError(f"prompt image slots={len(text_splits)-1}, image count={len(image_urls)}")
        content = [{"type": "text", "text": text_splits[0]}]
        for suffix, image_url in zip(text_splits[1:], image_urls):
            content.append({"type": "image_url", "image_url": {"url": image_url}})
            content.append({"type": "text", "text": suffix})
        return [{"role": "user", "content": content}]

    def get_answer(self, prompt, image_list):
        image_urls = [self.encode_image(path) for path in image_list]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.create_message(prompt, image_urls),
            temperature=self.temperature,
        )
        return response.choices[0].message.content

    def parse_stage(self, text):
        match = re.search(r"### RESULT ###\s*(\[.*?\])", text or "", re.DOTALL)
        return match.group(1) if match else ""

    def inference(self, images_input, instruction, Edit_Image, pure_image=None, system_prompt="", Metric="", first_stage_sys="", hints=""):
        input_images = list(images_input)
        if Metric == "VQ":
            return self.get_answer(prompt=system_prompt, image_list=[Edit_Image])

        if Metric == "URC":
            first_stage_prompt = first_stage_sys + "<image> <image>"
            compare_images = [pure_image, Edit_Image] if pure_image is not None else [input_images[0], Edit_Image]
            first_stage_response = self.get_answer(prompt=first_stage_prompt, image_list=compare_images)
            image_difference = self.parse_stage(first_stage_response)
            prompt = f"""
                1. Edit Instruction: {instruction},
                2. Source Image: <image>.
                3. Edited Image: <image>,
                4. Image Difference between Source Image and Edited Image: {image_difference}.
                """
            if hints:
                prompt += f"5. Hints: {hints}."
            return self.get_answer(prompt=system_prompt + prompt, image_list=[input_images[0], Edit_Image])

        prompt = f"""
                1. Edit Instruction: {instruction},
                2. Source Image: <image>,
                3. Edited Image: <image>.
                """
        if hints:
            prompt += f"4. Hints: {hints}."
        return self.get_answer(prompt=system_prompt + prompt, image_list=[input_images[0], Edit_Image])


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def import_metric_components(benchmark_root, metric):
    sys.path.insert(0, str(Path(benchmark_root)))
    if metric == "IF":
        from Prompt.Instruction_Following import Single_image_prompt
    elif metric == "IC":
        from Prompt.Identify_Consistecy import Single_image_prompt
    elif metric == "WA":
        from Prompt.World_Knowledge_Aware import Single_image_prompt
    elif metric == "URC":
        from Prompt.Unedited_Region_Consistency import Single_image_prompt, first_stage_prompt
        return Single_image_prompt, first_stage_prompt
    elif metric == "VQ":
        from Prompt.Visual_Quality import Single_image_prompt
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return Single_image_prompt, ""


def parse_response(response):
    if not response:
        return None
    try:
        return json.loads(response.strip())
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", response).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        text = re.sub(r",\s*([\]}])", r"\1", match.group())
        try:
            return json.loads(text)
        except Exception:
            return None
    return None


def language_result_root(save_root, language):
    return Path(save_root) / language


def evaluate_metric(args, metric, evaluator):
    system_prompt, first_stage_prompt = import_metric_components(args.benchmark_root, metric)
    result_root = language_result_root(args.save_root, args.language)
    metric_field = METRIC_FIELD[metric]
    lock = threading.Lock()

    for cfg in iter_task_configs(args.data_root, result_root, args.parts):
        save_dir = cfg["save_dir"]
        task_json = save_dir / f"{cfg['task_name']}.json"
        if not task_json.exists():
            print(f"[edit-compass-eval] missing generated json: {task_json}")
            continue

        data = read_json(task_json)
        output_json = save_dir / f"{cfg['task_name']}_{metric}.json"
        existing = []
        processed = set()
        if output_json.exists() and not args.overwrite:
            existing = [row for row in read_json(output_json) if row.get(f"eval_{metric}_status") == "Success"]
            processed = {row["save_id"] for row in existing}
            print(f"[edit-compass-eval] resume {cfg['task_name']} {metric}: {len(existing)} processed")

        save_data = list(existing)
        remaining = [
            item for item in data
            if item.get(metric) is True
            and item.get("save_id") not in processed
            and item.get("status") == "success"
            and os.path.exists(item.get("output_image_path", ""))
        ]
        if args.limit_per_task > 0:
            remaining = remaining[: args.limit_per_task]
        print(f"[edit-compass-eval] {cfg['part']}/{cfg['task_name']} {metric}: remaining={len(remaining)}")

        def worker(item):
            edited_path = item.get("output_image_path") or str(save_dir / f"{item['save_id']}.png")
            source_path = os.path.join(args.data_root, item["source_image"])
            instruction = item[f"prompt_{args.language}"]
            try:
                response = evaluator.inference(
                    images_input=[source_path],
                    instruction=instruction,
                    Edit_Image=edited_path,
                    system_prompt=system_prompt,
                    first_stage_sys=first_stage_prompt,
                    Metric=metric,
                    hints=item.get("hints", ""),
                )
                parsed = parse_response(response)
                if parsed and isinstance(parsed, dict) and "score" in parsed:
                    row = dict(item)
                    row[f"eval_{metric}_status"] = "Success"
                    row[metric_field] = parsed
                    with lock:
                        save_data.append(row)
                        write_json(str(output_json), save_data)
                else:
                    print(f"[edit-compass-eval] parse failed save_id={item.get('save_id')} metric={metric}: {response}")
            except Exception as exc:
                print(f"[edit-compass-eval] failed save_id={item.get('save_id')} metric={metric}: {exc}")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            list(tqdm(executor.map(worker, remaining), total=len(remaining), desc=f"{cfg['task_name']} {metric}"))
        write_json(str(output_json), save_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_root", default=str(DG2_ROOT / "third_party" / "Edit-Compass-and-EditReward-Compass"))
    parser.add_argument("--data_root", default=str(DG2_ROOT / "third_party" / "Edit-Compass-data"))
    parser.add_argument("--save_root", required=True, help="Root containing language folders, e.g. OUTPUT_ROOT/MODEL_NAME")
    parser.add_argument("--language", default="en")
    parser.add_argument("--parts", nargs="+", default=["Part1", "Part2"])
    parser.add_argument("--metrics", nargs="+", default=["IC", "URC", "VQ", "WA", "IF"], choices=list(METRIC_FIELD))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--api_key", default=os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "")
    parser.add_argument("--api_base", default=os.environ.get("API_BASE") or os.environ.get("OPENAI_BASE_URL") or "")
    parser.add_argument("--eval_model_name", default=os.environ.get("EDIT_COMPASS_EVAL_MODEL_NAME", "gemini-3.1-pro-preview"))
    parser.add_argument("--api_max_image_side", type=int, default=int(os.environ.get("EDIT_COMPASS_API_MAX_IMAGE_SIDE", "2048")))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_per_task", type=int, default=0)
    args = parser.parse_args()

    evaluator = OpenAICompatibleEvalWrapper(
        model_name=args.eval_model_name,
        api_key=args.api_key,
        api_base=args.api_base,
        max_image_side=args.api_max_image_side,
    )
    for metric in args.metrics:
        evaluate_metric(args, metric, evaluator)


if __name__ == "__main__":
    main()
