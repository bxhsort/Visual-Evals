import os
import torch


def resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_index = local_rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(device_index)
    return torch.device(f"cuda:{device_index}")
