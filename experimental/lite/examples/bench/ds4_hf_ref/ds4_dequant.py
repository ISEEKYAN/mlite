"""FP4/FP8 dequant helpers — VERBATIM copy of mlite's deepseek_v4/lite/checkpoint.py
helpers, so the HF reference dequantizes the real release with bit-identical math to
mlite (fairness: the only mlite-vs-HF difference is the forward kernels, not the load)."""
import math

import torch

_FP4_E2M1_TABLE = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype.is_floating_point:
        return scale.float()
    if scale.dtype == torch.uint8:
        return torch.pow(torch.tensor(2.0, dtype=torch.float32), scale.float() - 127.0)
    return scale.float()


def _expand_block_scale(scale: torch.Tensor, target_shape) -> torch.Tensor:
    target = tuple(int(dim) for dim in target_shape)
    while scale.ndim > len(target) and scale.shape[0] == 1:
        scale = scale.squeeze(0)
    while scale.ndim < len(target):
        scale = scale.unsqueeze(-1)
    if tuple(scale.shape) == target:
        return scale
    out = scale
    for dim, size in enumerate(target):
        if out.shape[dim] == size:
            continue
        repeat = math.ceil(size / out.shape[dim])
        out = out.repeat_interleave(repeat, dim=dim)
    slices = tuple(slice(0, size) for size in target)
    return out[slices]


def _unpack_fp4_e2m1_if_needed(tensor: torch.Tensor, target_shape) -> torch.Tensor:
    target = tuple(int(dim) for dim in target_shape)
    if (
        tensor.dtype != torch.int8
        or tensor.ndim != len(target)
        or tuple(tensor.shape[:-1]) != target[:-1]
        or tensor.shape[-1] * 2 != target[-1]
    ):
        return tensor.float()
    table = torch.tensor(_FP4_E2M1_TABLE, dtype=torch.float32, device=tensor.device)
    packed = tensor.view(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((table[low.long()], table[high.long()]), dim=-1).flatten(-2)


def dequantize_scaled_tensor(tensor: torch.Tensor, scale: torch.Tensor, shape) -> torch.Tensor:
    scale_f = _expand_block_scale(_scale_to_float(scale), shape)
    return _unpack_fp4_e2m1_if_needed(tensor, shape) * scale_f
