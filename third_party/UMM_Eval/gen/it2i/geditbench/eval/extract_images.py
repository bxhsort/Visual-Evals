import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset, load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", help="Local datasets.save_to_disk GEdit-Bench directory")
args = parser.parse_args()
dataset = load_from_disk(args.dataset_path)
### summerize the average resolution of input images
total_width = 0
total_height = 0
num_images = 0
for sample in tqdm(dataset):
    input_image_raw = sample['input_image_raw'].convert("RGB")
    # input_image_raw.save(f"input_images/{sample['key']}.png")
    width, height = input_image_raw.size
    total_width += width
    total_height += height
    num_images += 1
average_width = total_width / num_images
average_height = total_height / num_images
print(f"Average resolution of input images: {average_width} x {average_height}")
