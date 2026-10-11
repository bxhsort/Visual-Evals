# OneIG 本地评测权重

默认 checkpoint 根目录：

```text
/inspire/hdd3/project/agent-driven-world-model/wangdianyi-240107110022/qb-lim/BXH/DG2-Eval/ckpt
```

| 评测项 | 默认读取位置（相对于 checkpoint 根目录） |
| --- | --- |
| diversity | `dreamsim/`：`clip_vitb16_pretrain.pth.tar`、`dino_vitb16_pretrain.pth`、`open_clip_vitb16_pretrain.pth.tar`、`ensemble_lora/` |
| style：CSD | `CSD/checkpoint.pth`；不存在时依次查找同目录下的 `pytorch_model.bin`、`csd-vit-l.pth` |
| style：CSD 初始化 CLIP | `ViT-L-14.pt` |
| style：StyleEncoder | `OneIG-StyleEncoder/` |
| reasoning：图像编码器 | `LLM2CLIP-Openai-L-14-336/` |
| reasoning：文本编码器及 tokenizer | `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned/` |
| reasoning：图像预处理配置 | `clip-vit-large-patch14-336/` |

alignment 和 text 共用以下 Qwen 权重：

```text
/inspire/hdd3/project/agent-driven-world-model/public/models/public_models/Qwen/Qwen2.5-VL-7B-Instruct
```

路径已作为默认值写入代码，原来的生图和评测启动命令无需额外设置这些路径。
其他机器可用 `ONEIG_CKPT_ROOT` 覆盖 checkpoint 根目录。Qwen 路径优先采用
`QWEN_VL_MODEL_PATH`，其次采用 LongTextBench 的 `QWEN_OCR_MODEL_PATH`，最后采用上述公共路径。

缺失路径会明确报错，CSD 只在指定的下载目录中查找文件；DreamSim 指定同一下载目录作为
`cache_dir`，启动前检查三个基础权重和 `ensemble_lora/` 是否存在。Transformers 模型和配置采用
`local_files_only=True`，文件不完整时应补齐本地下载。DreamSim 库自身仍可能需要 Torch Hub
的 DINO 源代码，本适配不改变其代码获取方式。

DreamSim 仍使用 ensemble，CSD 的 checkpoint 加载格式和各项评分计算保持原样。
