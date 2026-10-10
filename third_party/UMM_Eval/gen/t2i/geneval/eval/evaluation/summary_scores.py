# Get results of evaluation

import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
args = parser.parse_args()
output_path = args.filename.replace('.jsonl', '_scores.txt')

# Load classnames
with open(os.path.join(os.path.dirname(__file__), "object_names.txt")) as cls_file:
    classnames = [line.strip() for line in cls_file]
    cls_to_idx = {"_".join(cls.split()): idx for idx, cls in enumerate(classnames)}

# Load results
df = pd.read_json(args.filename, orient="records", lines=True)

# -------- Collect print content --------
lines = []

def log(s=""):
    print(s)
    lines.append(str(s))

# Measure overall success
log("Summary")
log("=======")
log(f"Total images: {len(df)}")
log(f"Total prompts: {len(df.groupby('metadata'))}")
log(f"% correct images: {df['correct'].mean():.2%}")
log(f"% correct prompts: {df.groupby('metadata')['correct'].any().mean():.2%}")
log()

# By group
task_scores = []

log("Task breakdown")
log("==============")
for tag, task_df in df.groupby('tag', sort=False):
    task_scores.append(task_df['correct'].mean())
    log(f"{tag:<16} = {task_df['correct'].mean():.2%} "
        f"({task_df['correct'].sum()} / {len(task_df)})")
log()

log(f"Overall score (avg. over tasks): {np.mean(task_scores):.5f}")

# -------- Save to txt --------
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nSaved summary to: {output_path}")
