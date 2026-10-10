import base64
import os
import json
import argparse
import time
from io import BytesIO
from openai import OpenAI
from tqdm import tqdm
# from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
from eval_api_key import resolve_eval_api_key, TOKENHUB_BASE_URL


def load_prompts(prompts_json_path):
    with open(prompts_json_path, 'r') as f:
        return json.load(f)

def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

def resized_image_to_base64(image_path, target_size):
    try:
        with Image.open(image_path) as image:
            resized_image = image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            buffer = BytesIO()
            resized_image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except FileNotFoundError:
        print(f"File {image_path} not found.")
        return None

# @retry(wait=wait_exponential(multiplier=1, min=2, max=2), stop=stop_after_attempt(100))
def _should_retry(exception):
    error_text = str(exception).lower()
    retry_markers = [
        "internal error",
        "server_error",
        "rate limit",
        "timeout",
        "timed out",
        "connection",
        "temporarily unavailable",
        "invalid_prompt",
    ]
    return any(marker in error_text for marker in retry_markers)


def call_gpt(client, original_image_path, result_image_path, edit_prompt, edit_type, prompts, max_retries=100, retry_delay=1):
    with Image.open(original_image_path) as original_image:
        original_size = original_image.size

    original_image_base64 = image_to_base64(original_image_path)
    result_image_base64 = resized_image_to_base64(result_image_path, original_size)

    if not original_image_base64 or not result_image_base64:
        return {"error": "Image conversion failed"}

    prompt = prompts[edit_type]
    full_prompt = prompt.replace('<edit_prompt>', edit_prompt)

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=os.environ.get("IMGEDIT_EVAL_MODEL", "gpt-4o"),
                stream=False,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{original_image_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result_image_base64}"}}
                    ]
                }],
                temperature=0.0,
                max_tokens=2000,
            )
            return response
        except Exception as e:
            last_exception = e
            print(f"Error in calling GPT API (attempt {attempt}/{max_retries}): {e}")
            if attempt == max_retries or not _should_retry(e):
                raise

            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    raise last_exception

def process_single_item(client, key, item, result_img_folder, origin_img_root, prompts):
    result_img_name = f"{key}.png"
    result_img_path = os.path.join(result_img_folder, result_img_name)
    origin_img_path = os.path.join(origin_img_root, item['id'])
    edit_prompt = item['prompt']
    edit_type = item['edit_type']

    response = call_gpt(client, origin_img_path, result_img_path, edit_prompt, edit_type, prompts)
    return key, response.choices[0].message.content

def _atomic_json_dump(obj, path, indent=4):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    os.replace(tmp_path, path)

def process_json(client, edit_json, result_img_folder, origin_img_root, result_folder, num_threads, prompts):
    with open(edit_json, 'r', encoding="utf-8") as f:
        edit_infos = json.load(f)

    os.makedirs(result_folder, exist_ok=True)
    results_path = os.path.join(result_folder, 'imgedit_result.json')

    results = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                exists_results = json.load(f)
            for k, v in exists_results.items():
                if isinstance(v, dict) and "error" in v:
                    continue
                results[k] = v
        except Exception as e:
            print(f"[WARN] Failed to load existing results ({results_path}): {e}. Start fresh.")
            results = {}

    for k in list(results.keys()):
        if k in edit_infos:
            del edit_infos[k]

    if len(edit_infos) == 0:
        _atomic_json_dump(results, results_path)
        print("No pending items. Resume finished.")
        return

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_key = {
            executor.submit(process_single_item, client, key, item, result_img_folder, origin_img_root, prompts): key
            for key, item in edit_infos.items()
        }

        for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing edits"):
            key = future_to_key[future]
            try:
                k, result = future.result()
                results[k] = result
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                results[key] = {"error": str(e)}

            _atomic_json_dump(results, results_path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate image edits using GPT")
    parser.add_argument('--result_img_folder', type=str, required=True, help="Folder with subfolders of edited images")
    parser.add_argument('--api_key', required=False, default=None)
    parser.add_argument('--edit_json', type=str, required=True, help="Path to JSON file mapping keys to metadata")
    parser.add_argument('--origin_img_root', type=str, required=True, help="Root path where original images are stored")
    parser.add_argument('--result_folder', type=str, required=True, help="Root path where the results will be saved")
    parser.add_argument('--num_processes', type=int, default=32, help="Number of parallel threads")
    parser.add_argument('--prompts_json', type=str, required=True, help="JSON file containing prompts") 
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)

    client = OpenAI(
        api_key=resolve_eval_api_key(args.api_key),
        base_url=TOKENHUB_BASE_URL,
    )

    process_json(client, args.edit_json, args.result_img_folder, args.origin_img_root, args.result_folder, args.num_processes, prompts)

if __name__ == "__main__":
    main()
