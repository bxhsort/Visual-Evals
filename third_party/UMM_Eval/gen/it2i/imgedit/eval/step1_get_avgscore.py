import json
import argparse
import re
from typing import Optional
# def extract_scores_and_average(entry: str) -> float:
#     lines = entry.splitlines()
#     scores = []
#     for line in lines:
#         parts = line.strip().split(': ')
#         if len(parts) == 2 and parts[1].isdigit():
#             scores.append(int(parts[1]))
#     if scores:
#         return round(sum(scores) / len(scores), 2)
#     return None

def extract_scores_and_average(entry: str) -> Optional[float]:
    # Match patterns like: "Something: 4"
    # scores = re.findall(r":\s*(\d+)", entry)
    # Also handle markdown-bold scores like "**Something:** 4"
    scores = re.findall(r":\s*\**\s*(\d+)", entry)

    if scores:
        scores = [int(s) for s in scores]
        if len(scores) > 3:
            # keep the last three scores
            scores = scores[-3:]
        return round(sum(scores) / len(scores), 2)

    return None

def compute_averages(result_json_dict):
    result = {}
    for key, value in result_json_dict.items():
        avg = extract_scores_and_average(value)
        # judge 拒评/解析不出数字(如 extract 未抠图)视为模型失败, 记 0 分计入
        result[key] = avg if avg is not None else 0.0
    return result

def main():
    parser = argparse.ArgumentParser(description="Calculate the average score for each key and save it as a new JSON file")
    parser.add_argument('--result_json', type=str, required=True, help='Path of result_json json')
    parser.add_argument('--average_score_json', type=str, required=True, help='Path of average_score_json json')

    args = parser.parse_args()

    with open(args.result_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    error_keys = [k for k, v in data.items() if not isinstance(v, str)]
    if error_keys:
        print(f"[WARN] Skipping {len(error_keys)} non-string entries (likely API errors): {error_keys[:10]}")
        data = {k: v for k, v in data.items() if isinstance(v, str)}

    averaged_data = compute_averages(data)
    with open(args.average_score_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_data, f, indent=2)


if __name__ == '__main__':
    main()