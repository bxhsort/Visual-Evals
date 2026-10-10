"""
ms-swift PtEngine inference engine.

Mirrors the swift CLI command used to produce this dataset's *_response_* fields:

    swift infer --infer_backend pt --max_batch_size 24 --seed 42 \
                --temperature 0 --top_k 1 --top_p 1 \
                --repetition_penalty 1.05 --max_new_tokens 4096 \
                --enable_thinking true

Requires ms-swift>=4.0.0.
"""

import time

import torch
from swift import TransformersEngine, RequestConfig, InferRequest


def validate_linear_attention_backends(backends, require_fla):
    has_fla = any(name.startswith("fla.") for name in backends)
    if require_fla and not has_fla:
        raise RuntimeError(
            "Q-Judger is not using flash-linear-attention. The transformers "
            "fallback is prohibitively slow; install a compatible "
            "flash-linear-attention build or pass --no-require-fla for diagnostics."
        )


class MsSwiftJudge:
    def __init__(
        self,
        model_path,
        max_batch_size=12,
        max_new_tokens=4096,
        require_fla=True,
    ):
        self.engine = TransformersEngine(model_path, max_batch_size=max_batch_size)
        self.request_config = RequestConfig(
            max_tokens=max_new_tokens,
            temperature=0,
            top_k=1,
            top_p=1.0,
            repetition_penalty=1.05,
            seed=42,
        )

        # Enable Qwen3.5 thinking mode. The Q-Judger is a hybrid-thinking model whose
        # template DEFAULTS to non-thinking (emitting an empty `<think></think>`), which
        # changes scores vs. the official run. In ms-swift 4.3.x thinking is controlled by
        # Template.enable_thinking and per-request InferRequest.chat_template_kwargs
        # (template/base.py:_get_enable_thinking) — NOT template_meta.template_kwargs.
        # Set both for robustness; generate_batch also passes it per request.
        try:
            self.engine.default_template.enable_thinking = True
        except AttributeError:
            pass
        self.last_batch_metrics = {}

        linear_attention_backends = self._linear_attention_backends()
        print(
            "Qwen3.5 linear-attention backends: "
            + (", ".join(linear_attention_backends) or "not detected")
        )
        validate_linear_attention_backends(linear_attention_backends, require_fla)

    def _linear_attention_backends(self):
        backends = set()
        for module in self.engine.model.modules():
            kernel = getattr(module, "chunk_gated_delta_rule", None)
            if callable(kernel):
                module_name = getattr(kernel, "__module__", type(kernel).__module__)
                function_name = getattr(kernel, "__name__", type(kernel).__name__)
                backends.add(f"{module_name}.{function_name}")
        return sorted(backends)

    def set_max_batch_size(self, batch_size):
        self.engine.max_batch_size = batch_size

    def generate_batch(self, items):
        """
        Batch inference for multiple items.
        Each item: {"system_prompt": str, "user_text": str, "image": PIL.Image}
        Returns list of generated text strings.
        """
        infer_requests = []
        for item in items:
            messages = [
                {"role": "system", "content": item["system_prompt"]},
                {"role": "user", "content": item["user_text"]},
            ]
            infer_requests.append(
                InferRequest(
                    messages=messages,
                    images=[item["image"]],
                    chat_template_kwargs={"enable_thinking": True},
                )
            )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started_at = time.perf_counter()
        resp_list = self.engine.infer(
            infer_requests,
            self.request_config,
        )
        elapsed_seconds = time.perf_counter() - started_at

        completion_tokens = 0
        prompt_tokens = 0
        for response in resp_list:
            usage = getattr(response, "usage", None)
            completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0

        peak_cuda_allocated_bytes = 0
        peak_cuda_reserved_bytes = 0
        cuda_total_bytes = 0
        if torch.cuda.is_available():
            peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated()
            peak_cuda_reserved_bytes = torch.cuda.max_memory_reserved()
            cuda_total_bytes = torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory
        self.last_batch_metrics = {
            "elapsed_seconds": elapsed_seconds,
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
            "cuda_total_bytes": cuda_total_bytes,
        }

        return [r.choices[0].message.content for r in resp_list]
