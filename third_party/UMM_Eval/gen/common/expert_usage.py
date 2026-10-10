import math

import torch


def _new_layer_state(num_experts: int):
    return {
        "selected_counts": torch.zeros(num_experts, dtype=torch.long),
        "selected_weight_sums": torch.zeros(num_experts, dtype=torch.float64),
        "token_count": 0,
        "assignment_count": 0,
    }


def _entropy_metrics(values: torch.Tensor):
    total = float(values.sum().item())
    active_expert_count = int((values > 0).sum().item())
    active_expert_fraction = (
        active_expert_count / values.numel() if values.numel() > 0 else 0.0
    )
    if total <= 0.0 or active_expert_count == 0:
        return {
            "active_expert_count": active_expert_count,
            "active_expert_fraction": active_expert_fraction,
            "entropy": 0.0,
            "normalized_entropy": 0.0,
        }

    probs = values / total
    nonzero_probs = probs[probs > 0]
    entropy = float((-(nonzero_probs * torch.log(nonzero_probs))).sum().item())
    max_entropy = math.log(values.numel()) if values.numel() > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    return {
        "active_expert_count": active_expert_count,
        "active_expert_fraction": active_expert_fraction,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
    }


def _top_experts(values: torch.Tensor, top_n: int):
    if values.numel() == 0:
        return []
    top_n = min(top_n, values.numel())
    top_values, top_indices = torch.topk(values, k=top_n)
    result = []
    for expert_idx, value in zip(top_indices.tolist(), top_values.tolist()):
        scalar = float(value) if isinstance(value, float) else int(value)
        result.append({"expert_idx": int(expert_idx), "value": scalar})
    return result


class ExpertUsageTracker:
    def __init__(self, top_n: int = 8):
        self.top_n = top_n
        self.handles = []
        self.layer_specs = {}
        self.aggregate_stats = {}
        self.current_stats = {}
        self.capture_active = False

    @staticmethod
    def _is_moe_gate(module):
        if not hasattr(module, "top_k"):
            return False
        if hasattr(module, "n_routed_experts"):
            return True
        if hasattr(module, "num_experts") and hasattr(module, "experts"):
            return True
        return False

    @staticmethod
    def _num_experts_for_module(module):
        if hasattr(module, "n_routed_experts"):
            return int(module.n_routed_experts)
        if hasattr(module, "num_experts"):
            return int(module.num_experts)
        raise AttributeError("Unable to infer num_experts from module")

    @staticmethod
    def _extract_topk_from_router_logits(module, router_logits):
        routing_weights = torch.softmax(router_logits.detach().to(torch.float32), dim=-1)
        topk_weight, topk_idx = torch.topk(routing_weights, int(module.top_k), dim=-1)
        if getattr(module, "norm_topk_prob", False):
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        return topk_idx, topk_weight

    def _parse_output(self, module, output):
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return None, None

        first, second = output[0], output[1]
        num_experts = self._num_experts_for_module(module)

        if torch.is_tensor(first) and torch.is_tensor(second):
            if first.dtype in (torch.int32, torch.int64) and second.dtype.is_floating_point:
                return first, second
            if second.dim() == 2 and second.shape[-1] == num_experts:
                return self._extract_topk_from_router_logits(module, second)

        return None, None

    def attach(self, model):
        num_attached = 0
        for layer_name, module in model.named_modules():
            if not self._is_moe_gate(module):
                continue

            self.layer_specs[layer_name] = {
                "num_experts": self._num_experts_for_module(module),
                "top_k": int(module.top_k),
            }
            handle = module.register_forward_hook(self._make_hook(layer_name))
            self.handles.append(handle)
            num_attached += 1
        return num_attached

    def _make_hook(self, layer_name):
        def _hook(module, inputs, output):
            if not self.capture_active:
                return
            topk_idx, topk_weight = self._parse_output(module, output)
            if not torch.is_tensor(topk_idx) or not torch.is_tensor(topk_weight):
                return

            self._update_layer_stats(self.current_stats, layer_name, topk_idx, topk_weight)
            self._update_layer_stats(self.aggregate_stats, layer_name, topk_idx, topk_weight)

        return _hook

    def _update_layer_stats(self, stats_store, layer_name, topk_idx, topk_weight):
        spec = self.layer_specs[layer_name]
        layer_stats = stats_store.setdefault(
            layer_name, _new_layer_state(spec["num_experts"])
        )

        flat_idx = topk_idx.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        flat_weight = topk_weight.detach().reshape(-1).to(device="cpu", dtype=torch.float64)

        layer_stats["selected_counts"].index_add_(
            0, flat_idx, torch.ones_like(flat_idx, dtype=torch.long)
        )
        layer_stats["selected_weight_sums"].index_add_(0, flat_idx, flat_weight)
        layer_stats["token_count"] += int(topk_idx.shape[0])
        layer_stats["assignment_count"] += int(flat_idx.numel())

    def start_capture(self):
        self.current_stats = {}
        self.capture_active = True

    def finish_capture(self):
        self.capture_active = False
        return self._serialize(self.current_stats)

    def summarize_aggregate(self):
        return self._serialize(self.aggregate_stats)

    def _serialize(self, stats_store):
        layers = {}
        total_assignments = 0
        total_tokens = 0

        for layer_name, spec in self.layer_specs.items():
            layer_stats = stats_store.get(layer_name, _new_layer_state(spec["num_experts"]))
            count_sum = int(layer_stats["selected_counts"].sum().item())
            weight_sum = float(layer_stats["selected_weight_sums"].sum().item())
            count_share = (
                layer_stats["selected_counts"].to(torch.float64) / count_sum
                if count_sum > 0
                else torch.zeros(spec["num_experts"], dtype=torch.float64)
            )
            weight_share = (
                layer_stats["selected_weight_sums"] / weight_sum
                if weight_sum > 0.0
                else torch.zeros(spec["num_experts"], dtype=torch.float64)
            )

            layers[layer_name] = {
                "num_experts": spec["num_experts"],
                "top_k": spec["top_k"],
                "token_count": int(layer_stats["token_count"]),
                "assignment_count": int(layer_stats["assignment_count"]),
                "selected_counts": layer_stats["selected_counts"].tolist(),
                "selected_count_share": count_share.tolist(),
                "selected_weight_sums": layer_stats["selected_weight_sums"].tolist(),
                "selected_weight_share": weight_share.tolist(),
                "count_metrics": _entropy_metrics(layer_stats["selected_counts"].to(torch.float64)),
                "weight_metrics": _entropy_metrics(layer_stats["selected_weight_sums"]),
                "top_experts_by_count": _top_experts(
                    layer_stats["selected_counts"], self.top_n
                ),
                "top_experts_by_weight": _top_experts(
                    layer_stats["selected_weight_sums"], self.top_n
                ),
            }
            total_assignments += int(layer_stats["assignment_count"])
            total_tokens += int(layer_stats["token_count"])

        return {
            "num_layers": len(self.layer_specs),
            "total_layer_token_count": total_tokens,
            "total_layer_assignment_count": total_assignments,
            "layers": layers,
        }
