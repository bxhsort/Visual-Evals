# DG2 Bagel 评测

两台机器不互通时，只搬运 DG2 代码；Python 环境、Bagel 权重和 benchmark 数据都在
目标机重新安装或下载。不要复制 `.venv`、`.conda`，也不要复制当前机器上的 AFS
软链接。

## 1. 支持的 benchmark

- T2I：`geneval`、`dpgbench`、`longtextbench`、`cvtg_2k`、`tiifbench`、
  `oneigbench`、`wise`、`bizgeneval`、`qwenimagebench`
- 图像编辑：`imgedit`、`geditbench`、`rise`、`editcompass`
- 不包含：`igbench`、`u1.5bench`、`inhouseeval`

## 2. 搬运代码

当前机器执行：

```bash
cd /mnt/afs/baixuehai/tmp/DG2
bash scripts/setup/build_code_bundle.sh ../DG2-code.tar.gz
```

将 `DG2-code.tar.gz` 和 `DG2-code.tar.gz.sha256` 通过 U 盘、制品库或 Git 仓库带到
目标机，然后执行：

```bash
sha256sum -c DG2-code.tar.gz.sha256
tar -xzf DG2-code.tar.gz
cd DG2
```

压缩包只包含代码和小型 prompt，不包含环境、模型、benchmark 大文件和 AFS 绝对
软链接。

## 3. 安装 Bagel

DG2 的生成逻辑沿用 [Bagel 官方仓库](https://github.com/ByteDance-Seed/Bagel)。
在目标机按官方 README 准备 CUDA/PyTorch，并在 DG2 根目录创建生成环境：

```bash
conda create -p .venv python=3.10 -y
conda run -p .venv pip install -r requirements.txt
conda run -p .venv pip install flash_attn==2.5.8 --no-build-isolation
conda run -p .venv pip install openai
```

从 [BAGEL-7B-MoT](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT) 下载权重到
`models/BAGEL-7B-MoT/`：

```bash
mkdir -p models
.venv/bin/huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT \
  --local-dir models/BAGEL-7B-MoT
```

## 4. 安装 benchmark

每个 benchmark 的评测环境和额外模型以其官方 GitHub README 为准。DG2 不再提供
自动建环境或自动下载脚本。

| TASK | 官方代码/安装说明 | DG2 默认数据或模型位置 |
| --- | --- | --- |
| `geneval` | [GenEval](https://github.com/djghosh13/geneval) | `third_party/UMM_Eval/gen/t2i/geneval/eval/models/` |
| `dpgbench` | [ELLA/dpg_bench](https://github.com/TencentQQGYLab/ELLA/tree/main/dpg_bench) | `third_party/UMM_Eval/gen/t2i/dpgbench/eval/mplug_visual-question-answering_coco_large_en/` |
| `longtextbench` | [X-Omni](https://github.com/X-Omni-Team/X-Omni) | evaluator 与 `oneigbench` 共用 Qwen2.5-VL |
| `cvtg_2k` | [TextCrafter](https://github.com/NJU-PCALab/TextCrafter) | `third_party/UMM_Eval/gen/t2i/cvtg_2k/eval/{CVTG-2K,TextCrafter_Eval}/` |
| `tiifbench` | [TIIF-Bench](https://github.com/A113N-W3I/TIIF-Bench) | `third_party/UMM_Eval/gen/t2i/tiifbench/eval/TIIF-Bench/` |
| `oneigbench` | [OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark) | `third_party/UMM_Eval/gen/t2i/oneigbench/eval/OneIG-Benchmark/` |
| `wise` | [WISE](https://github.com/PKU-YuanGroup/WISE) | prompt 已在 DG2 中；当前 wrapper 使用 legacy 协议 |
| `bizgeneval` | [BizGenEval](https://github.com/microsoft/BizGenEval) | prompt 已在 DG2 中 |
| `qwenimagebench` | [Qwen-Image-Bench](https://github.com/QwenLM/Qwen-Image-Bench) | `third_party/UMM_Eval/gen/t2i/qwenimagebench/eval/models/q_judger/` |
| `imgedit` | [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit) | `third_party/UMM_Eval/gen/it2i/imgedit/eval/Benchmark/` |
| `geditbench` | [GEdit-Bench](https://github.com/stepfun-ai/GEdit-Bench) | `third_party/UMM_Eval/gen/it2i/geditbench/eval/GEdit-Bench/` |
| `rise` | [RISEBench](https://github.com/PhoenixZ810/RISEBench) | `third_party/UMM_Eval/gen/it2i/rise/eval/data/` |
| `editcompass` | [Edit-Compass](https://github.com/bxhsort/Edit-Compass-and-EditReward-Compass) | `third_party/Edit-Compass-data/` |

本地 judge/model 的下载入口：

- DPG mPLUG：`damo/mplug_visual-question-answering_coco_large_en`（ModelScope）
- OneIG/LongText OCR：`Qwen/Qwen2.5-VL-7B-Instruct`（Hugging Face）
- Q-Judger：`Qwen/Qwen-Image-Bench`（Hugging Face）

EditCompass 还需要官方评分 prompt 和数据集。数据集需要先在 Hugging Face 页面接受
访问条件：

```bash
git clone https://github.com/bxhsort/Edit-Compass-and-EditReward-Compass.git \
  third_party/Edit-Compass-and-EditReward-Compass

.venv/bin/huggingface-cli login
.venv/bin/huggingface-cli download --repo-type dataset DogNeverSleep/Edit-Compass \
  --local-dir third_party/Edit-Compass-data
```

当前 DG2 的 Bagel EditCompass 适配器支持 `Part1`、`Part2`，以及 `en`、`cn` 两种
评测语言；官方新增的 `Part3` 至 `Part6` 尚未接入。

可以为不同 evaluator 分别创建 conda 环境。创建后，把环境路径写入下一节的
`config/eval.env`；具体依赖版本直接跟随上表对应官方仓库。

## 5. 配置与运行

```bash
cp config/eval.env.example config/eval.env
```

至少检查 `MODEL_PATH`、`BAGEL_ENV`、`GPU_LIST` 和所运行 benchmark 的 `*_EVAL_ENV`。
需要 API judge 的任务还要在 shell 或任务平台中设置 `API_KEY` 和 API 地址，不要把
密钥写进仓库。

T2I：

```bash
TASK=geneval bash eval_t2i_go.sh
TASK=oneigbench bash eval_t2i_go.sh
```

图像编辑：

```bash
TASK=imgedit bash eval_go.sh
TASK=geditbench bash eval_go.sh
TASK=rise bash eval_go.sh
TASK=editcompass PARTS="Part1 Part2" EDIT_COMPASS_LANGUAGE=en bash eval_go.sh
```

只生图或只评分：

```bash
RUN_GENERATION=1 RUN_EVAL=0 TASK=geneval bash eval_t2i_go.sh
RUN_GENERATION=0 RUN_EVAL=1 TASK=geneval bash eval_t2i_go.sh
```

结果默认写入 `outputs/`。
