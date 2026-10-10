import argparse
import ast
import os
import re

import megfile
import pandas as pd
from rich.console import Console
from rich.table import Table


SOURCE_ID = {
    "anime": "Anime_Stylization",
    "human": "Portrait",
    "object": "General_Object",
    "text": "Text_Rendering",
    "reasoning": "Knowledge_Reasoning",
    "multilingualism": "Multilingualism",
}
KEYS = ["alignment", "diversity", "reasoning", "style", "text"]

md_content = []
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(description="OneIG-Bench fine-grained analysis.")
    parser.add_argument("--mode", type=str, default="EN", choices=["EN", "ZH"])
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    return parser.parse_args()


def save_to_markdown(title, categories, raw_data):
    md_content.append(f"\n## {title}\n")
    header = "| Model | " + " | ".join(categories) + " |"
    split = "|------|" + "|".join(["---" for _ in categories]) + "|"
    md_content.append(header)
    md_content.append(split)

    table = Table(title=title, show_lines=True)
    table.add_column("Model", justify="left", style="bold")
    for cat in categories:
        table.add_column(cat, justify="center")

    for model, scores in raw_data.items():
        row = [model]
        for cat in categories:
            score = scores.get(cat, {}).get("score")
            val = f"{score:.3f}" if score is not None else "-"
            row.append(val)
        md_content.append(f"| {' | '.join(row)} |")
        table.add_row(*row)
    console.print(table)


def load_metric_csv(result_dir, pattern):
    matches = megfile.smart_glob(os.path.join(result_dir, pattern))
    if not matches:
        return None
    # Filenames embed an ISO-like timestamp (..._YYYY-MM-DD_HH-MM-SS.csv),
    # so lexicographic max == latest run. Avoids picking a stale empty CSV
    # when multiple runs coexist in the same result_dir.
    return pd.read_csv(sorted(matches)[-1], dtype=str)


def parse_text_metric_value(raw_value):
    cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", raw_value)
    return ast.literal_eval(cleaned)


def infer_model_names(result_dir, mode):
    patterns = [
        f"alignment_prompt_score_{mode}*.csv",
        f"diversity_prompt_score_{mode}*.csv",
        f"reasoning_prompt_score_{mode}*.csv",
        f"style_prompt_score_{mode}*.csv",
        f"text_prompt_score_{mode}*.csv",
    ]
    for pattern in patterns:
        data = load_metric_csv(result_dir, pattern)
        if data is not None:
            return [col for col in data.columns if col != "Unnamed: 0"]
    return []


def compute_scores(base_df, data, model_name, filter_fn):
    scores = [float(row[model_name]) for _, row in data.iterrows() if filter_fn(base_df, row)]
    return {"score": sum(scores) / len(scores) if scores else None, "num": len(scores)}


def calc_overall_score(overall_scores, model, key, data):
    all_scores = []
    for value in data.values():
        if isinstance(value, dict) and value.get("score") is not None:
            all_scores.append(value["score"])
    overall_scores[model][key] = sum(all_scores) / len(all_scores) if all_scores else None


def load_overall_scores(result_dir, mode, model_names):
    overall_scores = {model: {} for model in model_names}
    metric_specs = {
        "alignment": (f"alignment_score_{mode}*.csv", "alignment"),
        "diversity": (f"diversity_score_{mode}*.csv", "total average"),
        "reasoning": (f"reasoning_score_{mode}*.csv", "reasoning"),
        "style": (f"style_score_{mode}*.csv", "style"),
        "text": (f"text_score_{mode}*.csv", "text score"),
    }
    for metric_name, (pattern, column_name) in metric_specs.items():
        data = load_metric_csv(result_dir, pattern)
        if data is None:
            continue
        for model_name in model_names:
            row = data[data["Unnamed: 0"] == model_name]
            if row.empty or pd.isna(row.iloc[0].get(column_name)):
                overall_scores[model_name][metric_name] = None
            else:
                overall_scores[model_name][metric_name] = float(row.iloc[0][column_name])
    return overall_scores


def append_overall_markdown(overall_scores, model_names):
    md_content.append("\n## 📊 Overall Scores (Average)\n")
    overall_cols = ["Alignment ↑", "Diversity ↑", "Reasoning ↑", "Style ↑", "Text ↑"]
    key_map = {
        "Alignment ↑": "alignment",
        "Diversity ↑": "diversity",
        "Reasoning ↑": "reasoning",
        "Style ↑": "style",
        "Text ↑": "text",
    }
    header = "| Model | " + " | ".join(overall_cols) + " |"
    split = "|------|" + "|".join(["---" for _ in overall_cols]) + "|"
    md_content.append(header)
    md_content.append(split)

    table = Table(title="Overall Scores (Average)", show_lines=True)
    table.add_column("Model", justify="left", style="bold")
    for col in overall_cols:
        table.add_column(col, justify="center")

    for model in model_names:
        row = [model]
        for col in overall_cols:
            score = overall_scores[model].get(key_map[col])
            row.append(f"{score:.3f}" if score is not None else "-")
        md_content.append(f"| {' | '.join(row)} |")
        table.add_row(*row)
    console.print(table)


def handle_prompt_based_metric(result_dir, mode, base_df, model_names, overall_scores, key):
    result = {}
    data = load_metric_csv(result_dir, f"{key}_prompt_score_{mode}*.csv")
    if data is None:
        return
    for model_name in model_names:
        result[model_name] = {}
        for tag in ["short", "middle", "long"]:
            result[model_name][tag] = compute_scores(
                base_df,
                data,
                model_name,
                lambda df_, row: df_[
                    (df_["category"] == SOURCE_ID[row["Unnamed: 0"].split("_")[0]])
                    & (df_["id"] == row["Unnamed: 0"].split("_")[1])
                ]["prompt_length"].iloc[0] == tag,
            )
        for tag in ["T, P", "NP"]:
            result[model_name][tag] = compute_scores(
                base_df,
                data,
                model_name,
                lambda df_, row: df_[
                    (df_["category"] == SOURCE_ID[row["Unnamed: 0"].split("_")[0]])
                    & (df_["id"] == row["Unnamed: 0"].split("_")[1])
                ]["type"].iloc[0] == tag,
            )
    save_to_markdown(f"{key.capitalize()} Model Scores", ["NP", "T, P", "short", "middle", "long"], result)
    for model_name in model_names:
        calc_overall_score(overall_scores, model_name, key, result[model_name])


def handle_reasoning(result_dir, mode, base_df, model_names, overall_scores):
    result = {}
    subjects = ["geography", "computer science", "biology", "mathematics", "physics", "chemistry", "common sense"]
    data = load_metric_csv(result_dir, f"reasoning_prompt_score_{mode}*.csv")
    if data is None:
        return
    for model_name in model_names:
        result[model_name] = {}
        for subject in subjects:
            scores = [
                float(row[model_name])
                for _, row in data.iterrows()
                if base_df[(base_df["category"] == SOURCE_ID["reasoning"]) & (base_df["id"] == row["Unnamed: 0"])]["class"].iloc[0] == subject
            ]
            result[model_name][subject] = {"score": sum(scores) / len(scores) if scores else None, "num": len(scores)}
    save_to_markdown("Reasoning Model Scores", subjects, result)
    for model_name in model_names:
        calc_overall_score(overall_scores, model_name, "reasoning", result[model_name])


def handle_style(result_dir, mode, model_names, overall_scores):
    style_types = {
        "traditional": ["abstract_expressionism", "art_nouveau", "baroque", "chinese_ink_painting", "cubism", "fauvism", "impressionism", "line_art", "minimalism", "pointillism", "pop_art", "rococo", "ukiyo-e"],
        "media": ["clay", "crayon", "graffiti", "lego", "comic", "pencil_sketch", "stone_sculpture", "watercolor"],
        "anime": ["celluloid", "chibi", "cyberpunk", "ghibli", "impasto", "pixar", "pixel_art", "3d_rendering"],
    }
    result = {}
    data = load_metric_csv(result_dir, f"style_style_score_{mode}*.csv")
    if data is None:
        return
    for model_name in model_names:
        result[model_name] = {}
        row = data[data["Unnamed: 0"] == model_name]
        if row.empty:
            continue
        row = row.iloc[0]
        for style_cat, styles in style_types.items():
            flat_scores = [float(row[style]) for style in styles if pd.notna(row.get(style))]
            result[model_name][style_cat] = {"score": sum(flat_scores) / len(flat_scores) if flat_scores else None}
    save_to_markdown("Style Model Scores", list(style_types.keys()), result)
    for model_name in model_names:
        calc_overall_score(overall_scores, model_name, "style", result[model_name])


def handle_text(result_dir, mode, base_df, model_names, overall_scores):
    metrics = ["ED", "CR", "WAC"]
    lengths = ["short", "middle", "long"]
    result = {}
    data = load_metric_csv(result_dir, f"text_prompt_score_{mode}*.csv")
    if data is None:
        return
    for model_name in model_names:
        result[model_name] = {}
        for length in lengths:
            filtered = [
                parse_text_metric_value(row[model_name])
                for _, row in data.iterrows()
                if pd.notna(row[model_name])
                and base_df[(base_df["category"] == SOURCE_ID["text"]) & (base_df["id"] == row["Unnamed: 0"])]["prompt_length"].iloc[0] == length
            ]
            if filtered:
                avg = [sum(metric) / len(metric) for metric in zip(*filtered)]
                for idx, metric_name in enumerate(metrics):
                    result[model_name][f"{length}_{metric_name}"] = {"score": avg[idx]}
    cols = [f"{length}_{metric_name}" for metric_name in metrics for length in lengths]
    save_to_markdown("Text Model Scores", cols, result)
    for model_name in model_names:
        calc_overall_score(overall_scores, model_name, "text", result[model_name])


def main():
    args = parse_args()
    base_df = pd.read_csv(args.data_file, dtype=str)
    model_names = infer_model_names(args.result_dir, args.mode)
    overall_scores = load_overall_scores(args.result_dir, args.mode, model_names)

    md_content.append("# OneIG-Bench Evaluation Results\n")
    md_content.append(f"**Mode**: {args.mode}\n")
    append_overall_markdown(overall_scores, model_names)

    for key in KEYS:
        if key in ["alignment", "diversity"]:
            handle_prompt_based_metric(args.result_dir, args.mode, base_df, model_names, overall_scores, key)
        elif key == "reasoning":
            handle_reasoning(args.result_dir, args.mode, base_df, model_names, overall_scores)
        elif key == "style":
            handle_style(args.result_dir, args.mode, model_names, overall_scores)
        elif key == "text":
            handle_text(args.result_dir, args.mode, base_df, model_names, overall_scores)

    with open(args.output_md, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(md_content))

    print(f"\n✅ All results saved to {args.output_md}")


if __name__ == "__main__":
    main()
