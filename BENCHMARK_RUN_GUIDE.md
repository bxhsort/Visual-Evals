# Visual-Evals 生图与评测启动指南

本文按仓库当前启动脚本整理，更新于 2026-10-11。每个 benchmark 分为「生图」和「评测」两部分，两个阶段单独启动。文中的安装命令是操作说明，不会自动执行。

## 使用说明

**用户必须填写实验名称 `EXPERIMENT_NAME`，当前代码没有为它设置默认值；未提供时会报错。** 本文所有启动命令都写了 `EXPERIMENT_NAME=baseline` 作为示例，请把 `baseline` 替换为实际实验名称，例如 `bagel_baseline`。名称不要包含路径分隔符。

`baseline` 也是文生图入口中 `MODEL_NAME` 的默认值，但 `MODEL_NAME` 和 `EXPERIMENT_NAME` 是两个不同的参数，前者不会自动填入后者。

**同一实验的生图和评测必须使用相同的 `EXPERIMENT_NAME`。** 例如用 `EXPERIMENT_NAME=bagel_baseline` 生图，评测时也填写 `bagel_baseline`，结果会保存到 `outputs/bagel_baseline/<setting>/<benchmark>/`。已有图片的实验应填写生图时的名称，不能另外起一个名称。

其余参数沿用已有实验配置，不再增加通用的环境激活和参数导出步骤。评测已有图片时，模型路径、模型名称、生成参数、语言与 prompt 版本也保持和生图时一致。若 `config/eval.env` 中写死了实验名称，请同步修改该项，避免它覆盖命令中的名称。

默认示例均使用顶层入口；直接运行第三方 `launch.sh` 的默认分辨率可能不同。

当前文生图顶层入口默认分辨率均为 1024，步数 50，CFG 4.0，时间偏移 3.0，seed 42。`NUM_IMAGES` 按各 benchmark 分别设置，避免把一个 benchmark 的图片数带到另一个。TIIF 和 WISE 的 `BATCH_SIZE` 必须为 1；其余文生图任务顶层默认是 4。请在新终端执行，不要混入之前导出的 `OUTPUT_FOLDER`、`RESULT_ROOT_DIR`、`OUTPUT_DIR` 等旧路径变量。

| 阶段 / benchmark | 当前脚本使用的环境 | 评测方式 |
| --- | --- | --- |
| 所有生图 | bagel | BAGEL 本地生成 |
| GenEval | geneval | 本地目标检测与 CLIP |
| DPG-Bench | geneval | 本地 mPLUG VQA |
| LongTextBench | bagel | 本地 Qwen2.5-VL OCR |
| CVTG-2K | textcrafter_eval | 本地 OCR、CLIP、VQA、美学评分 |
| OneIG-Bench | bagel | 本地多模型评分 |
| TIIF-Bench、WISE、BizGenEval | geneval | API judge |
| QwenImageBench | geneval | 默认本地 Q-Judger |
| ImgEdit、RISE、EditCompass | geneval | API judge |
| GEdit-Bench | bagel | 默认 GPT-4.1 API / VIEScore |

各启动脚本会激活上表中的环境。API 评测前，通过 shell 或任务平台设置对应密钥；本文不保存密钥。表中的环境名称是现有脚本行为，除已确认配置好的 GenEval 外，仍需确认服务器中对应依赖和数据已经就绪。

结果目录按以下三层保存，启动日志会打印实际路径：

```text
outputs/<EXPERIMENT_NAME>/<setting>/<benchmark>/
```

`setting` 包含生成参数和配置标识；GenEval 还包含 prompt 版本。实际开启编辑 thinking 时，第三层带 `_thinking` 后缀。当前统一文生图生成器不支持 thinking，本文示例按未开启 thinking 的情况编写。只生图和只评测的开关不改变实验目录。

## GenEval

### 生图

环境：bagel。默认配置：1024、每个 prompt 4 张图、PE 长 prompt（`long`），共 553 个 prompt。

短 prompt 文件是 `eval/gen/geneval/prompts/evaluation_metadata.jsonl`；PE 长 prompt 文件是 `eval/gen/geneval/prompts/evaluation_metadata_long.jsonl`。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=geneval GPU_LIST="0,1,2,3,4,5,6,7" \
GENEVAL_PROMPT_MODE=long GENEVAL_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

使用短 prompt 时把 `long` 改为 `short`；两个版本都生成时改为 `both`，会分别保存结果。

### 评测

GenEval 环境已经配置好。**每次启动机器后，先激活 geneval，再按 mmcv → mmdetection 的顺序执行以下两次 editable 安装。** 安装完成后回到评测仓库。

```bash
source activate geneval
export PATH=/opt/conda/envs/geneval/bin:$PATH

cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/hdd/geneval/mmcv
pip install -e .

cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/hdd/geneval/mmdetection
pip install -e .

cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=geneval GPU_LIST=0 EVAL_GPUS=1 \
GENEVAL_PROMPT_MODE=long GENEVAL_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```

评测的 `GENEVAL_PROMPT_MODE` 必须与生图一致。短 prompt 改为 `short`；两个版本都已生成则可用 `both`。

## DPG-Bench

### 生图

环境：bagel。默认配置：1024、每个 prompt 4 张图，按 2×2 拼图保存；默认 prompt 文件是 `third_party/UMM_Eval/gen/t2i/dpgbench/eval/prompts.jsonl`。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=dpgbench GPU_LIST="0,1,2,3,4,5,6,7" \
DPGBENCH_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

### 评测

环境：geneval。评测前先使用 geneval 的 Python 安装以下指定版本：

```bash
/opt/conda/envs/geneval/bin/python -m pip install --no-deps "gast==0.5.4"
/opt/conda/envs/geneval/bin/python -m pip install \
  --no-deps --no-build-isolation "fairseq==0.12.2"
```

```b
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=dpgbench GPU_LIST="0,1" EVAL_GPUS=2 \
DPGBENCH_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```

## LongTextBench

### 生图

环境：bagel。默认配置：1024、每个 prompt 独立生成 4 张图、英文 `en`。英文和中文各 160 个 prompt；默认文件位于 `third_party/UMM_Eval/gen/t2i/longtextbench/eval/` 下的 `text_prompts.jsonl` 和 `text_prompts_zh.jsonl`。

图片数量和命名已按 [LongText-Bench 官方说明](https://huggingface.co/datasets/X-Omni/LongText-Bench/blob/main/README.md)对齐：默认 `NUM_IMAGES=4`，4 张图分别保存到 `samples/{prompt_id}_{repeat_id}.png`，repeat_id 为 1～4，每张都参与 OCR 评分。每种语言 160 个 prompt，共 640 张评分图片；拼图另外保存在 `grids`，不参与评分。因此命令不再显式填写 `NUM_IMAGES`。

同步本次代码后使用以下命令。此前的单图结果需要重新生成四张；新保存格式会使用新的实验配置目录，不复用旧单图图片和分数。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=longtextbench GPU_LIST="0,1,2,3,4,5,6,7" \
LONGTEXTBENCH_LANGS=en LONGTEXTBENCH_IMAGE_SIZE=1024 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

中英文都生成时，将语言改为 `LONGTEXTBENCH_LANGS="en zh"`。

### 评测

环境：bagel。四张独立图片生成完成后，使用以下只评测命令；默认按每个 prompt 四张图核对覆盖情况。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=longtextbench GPU_LIST=0 EVAL_GPUS=1 \
LONGTEXTBENCH_LANGS=en LONGTEXTBENCH_IMAGE_SIZE=1024 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```

若生成了中英文，评测也使用 `LONGTEXTBENCH_LANGS="en zh"`。

## CVTG-2K

### 生图

官方 [TextCrafter 批量生成脚本](https://github.com/NJU-PCALab/TextCrafter/blob/main/TextCrafter_Flux/eval.py)对每条数据生成并保存 1 张图。本库默认 `NUM_IMAGES=1`，图片数量一致，因此生图和评测命令均省略该参数。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=cvtg_2k GPU_LIST="0,1,2,3,4,5,6,7" \
CVTG_2K_IMAGE_SIZE=1024 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

### 评测

环境：textcrafter_eval。OCR 固定使用 CPU，CLIPScore、VQAScore、美学评分继续使用 GPU。评测需要官方入口：
```text
third_party/UMM_Eval/gen/t2i/cvtg_2k/eval/TextCrafter_Eval/unified_metrics_eval.py
```

评测使用单个进程，避免多个进程重复计算同一批图片。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=cvtg_2k GPU_LIST=0 EVAL_GPUS=1 RESIZE_TO= \
CVTG_2K_IMAGE_SIZE=1024 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```

上面命令面向新实验布局。此前旧目录的图片需要单独使用第三方入口，并明确设置 `RESULT_ROOT_DIR` 和 `OUTPUT_FOLDER`；不要把旧目录参数混入顶层入口。

## TIIF-Bench

### 生图

环境：bagel。默认配置：1024、每个 prompt 1 张图、batch size 1、`testmini` 子集。
```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=tiifbench GPU_LIST="0,1,2,3,4,5,6,7" \
TIIFBENCH_SPLIT=testmini TIIFBENCH_IMAGE_SIZE=1024 NUM_IMAGES=1 BATCH_SIZE=1 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

### 评测

环境：geneval。默认配置的 API judge 为 `gpt-4o`。启动前提供可用的 API 密钥；评分 prompt 位于上述数据根目录的 `data/testmini_eval_prompts`。`testmini` 结果应标注为该子集的结果。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=tiifbench GPU_LIST=0 EVAL_GPUS=1 \
TIIFBENCH_SPLIT=testmini TIIFBENCH_IMAGE_SIZE=1024 NUM_IMAGES=1 BATCH_SIZE=1 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```

## OneIG-Bench

### 生图

环境：bagel。默认配置：1024、每个 prompt 4 张图、2×2 拼图、中英文 `en zh`。metadata 位于 `third_party/UMM_Eval/gen/t2i/oneigbench/eval/OneIG-Benchmark`。英文 1120 个 prompt，中文 1320 个 prompt，图片使用 WebP 保存。

```bash
cd /inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/DG2
EXPERIMENT_NAME=baseline TASK=oneigbench GPU_LIST="0,1,2,3,4,5,6,7" \
ONEIGBENCH_LANGS="en zh" ONEIGBENCH_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=1 RUN_EVAL=0 bash eval_t2i_go.sh
```

### 评测

继续使用 bagel，不切换到 geneval。已经生成图片时，只执行本节。

EXPERIMENT_NAME=baseline TASK=oneigbench GPU_LIST=0 EVAL_GPUS=1 \
ONEIGBENCH_LANGS="en zh" ONEIGBENCH_IMAGE_SIZE=1024 NUM_IMAGES=4 \
RUN_GENERATION=0 RUN_EVAL=1 bash eval_t2i_go.sh
```


