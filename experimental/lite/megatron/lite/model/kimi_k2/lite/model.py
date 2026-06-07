"""Kimi K2 lite native model."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.mla import MultiLatentAttention
from megatron.lite.primitive.modules.router import SigmoidTopKRouter
from megatron.lite.primitive.ops.cross_entropy import vocab_parallel_cross_entropy
from megatron.lite.primitive.ops.linear_cross_entropy import linear_cross_entropy
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.parallel import (
    ColumnParallelLinear,
    ParallelState,
    RowParallelLinear,
    VocabParallelEmbedding,
    VocabParallelOutput,
    build_pipeline_chunk_layout,
    gather_from_sequence_parallel,
    reduce_scatter_to_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.utils import build_fp8_recipe

try:
    from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl as _mcore_swiglu
except Exception:  # pragma: no cover - local static envs may not have Megatron-Core.
    _mcore_swiglu = None

_SP_GRAD_SUFFIXES: tuple[str, ...] = (
    ".input_layernorm.weight",
    ".self_attention.linear_q_down_proj.weight",
    ".self_attention.linear_q_up_proj.linear.layer_norm_weight",
    ".self_attention.linear_kv_down_proj.weight",
    ".self_attention.linear_kv_up_proj.linear.layer_norm_weight",
    ".mlp_norm.weight",
    ".mlp.gate_up.linear.layer_norm_weight",
    ".moe.router.gate.weight",
    ".norm.weight",
)


def _collect_sp_grad_params(model: nn.Module) -> list[nn.Parameter]:
    return [
        param
        for name, param in model.named_parameters()
        if any(name.endswith(suffix) for suffix in _SP_GRAD_SUFFIXES) or name == "norm.weight"
    ]


def _swiglu(x: torch.Tensor) -> torch.Tensor:
    if _mcore_swiglu is not None and x.is_cuda:
        return _mcore_swiglu(x, None, False, False)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return F.silu(x1) * x2


class DenseMLP(nn.Module):
    def __init__(self, config: KimiK2Config, ps: ParallelState):
        super().__init__()
        self.gate_up = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size * 2,
            ps,
            bias=False,
            normalization="RMSNorm",
            eps=config.rms_norm_eps,
        )
        self.down = RowParallelLinear(config.intermediate_size, config.hidden_size, ps, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(_swiglu(self.gate_up(x)))


class SharedExpert(nn.Module):
    def __init__(self, config: KimiK2Config, ps: ParallelState):
        super().__init__()
        self.ps = ps
        ffn = config.shared_expert_intermediate_size
        self.gate_up = _LocalLinear(config.hidden_size, ffn * 2 // ps.tp_size)
        self.down = _LocalLinear(ffn // ps.tp_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = x.dim() == 2
        if squeeze_batch:
            x = x.unsqueeze(1)
        full_x = gather_from_sequence_parallel(x, self.ps)
        partial_out = self.down(_swiglu(self.gate_up(full_x)))
        out = reduce_scatter_to_sequence_parallel(partial_out, self.ps)
        return out.squeeze(1) if squeeze_batch else out


class _LocalLinear(nn.Module):
    """TE linear without built-in TP collectives; weight remains TP-local."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MoELayer(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        ps: ParallelState,
        *,
        use_deepep: bool,
        router_bias_rate: float,
        fp8: bool,
        moe_act_recompute: bool,
    ):
        super().__init__()
        if fp8:
            raise NotImplementedError("Kimi K2 lite MoE fp8 training is not implemented yet.")
        self.router = SigmoidTopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=True,
            use_pre_softmax=True,
            persistent_expert_bias=True,
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            use_mcore_grouped_gemm=True,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        self.shared_expert = SharedExpert(config, ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape

        scores, indices = self.router(x)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x, scores, indices)
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        routed_out = self.dispatcher.combine(expert_out)
        shared_out = self.shared_expert(x)
        output = routed_out.view(input_shape)
        output += shared_out
        return output.to(x.dtype)


class KimiK2Layer(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        ps: ParallelState,
        layer_idx: int,
        *,
        use_deepep: bool = False,
        router_bias_rate: float = 0.0,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        use_thd: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = MultiLatentAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            ps=ps,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            use_thd=use_thd,
        )
        if config.is_moe_layer(layer_idx):
            self.mlp_norm: nn.Module | None = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.moe: MoELayer | None = MoELayer(
                config,
                ps,
                use_deepep=use_deepep,
                router_bias_rate=router_bias_rate,
                fp8=fp8,
                moe_act_recompute=moe_act_recompute,
            )
            self.mlp: DenseMLP | None = None
        else:
            self.mlp_norm = None
            self.moe = None
            self.mlp = DenseMLP(config, ps)

    def forward(self, x: torch.Tensor, packed_seq_params=None) -> torch.Tensor:
        x = x + self.self_attention(self.input_layernorm(x), packed_seq_params=packed_seq_params)
        if self.moe is not None:
            assert self.mlp_norm is not None
            mlp_input = self.mlp_norm(x)
            return x + self.moe(mlp_input)
        assert self.mlp is not None
        return x + self.mlp(x)


def _temperature_to_float(temperature: float | torch.Tensor) -> float:
    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("KimiK2Model supports scalar temperature only.")
        return float(temperature.detach().float().item())
    return float(temperature)


def _apply_attention_backend_override(backend: str | None) -> None:
    if backend in (None, "flash"):
        backend = "fused"
    env = {
        "auto": ("1", "1", "1"),
        "flash": ("1", "0", "0"),
        "fused": ("0", "1", "0"),
        "unfused": ("0", "0", "1"),
        "local": ("0", "0", "1"),
    }.get(backend)
    if env is None:
        raise ValueError(
            "attention_backend_override must be one of "
            "{'auto', 'flash', 'fused', 'unfused', 'local'}"
        )
    os.environ["NVTE_FLASH_ATTN"], os.environ["NVTE_FUSED_ATTN"], os.environ["NVTE_UNFUSED_ATTN"] = env


class KimiK2Model(nn.Module):
    def __init__(
        self,
        config: KimiK2Config,
        train_config,
        ps: ParallelState,
        *,
        vpp_chunk_id: int | None = None,
        router_bias_rate: float = 0.0,
        use_thd: bool = False,
        hf_path: str = "",
        attention_backend_override: str | None = None,
    ):
        super().__init__()
        del hf_path
        _apply_attention_backend_override(attention_backend_override)
        self.config = config
        self.train_config = train_config
        self.ps = ps
        self._input_tensor: torch.Tensor | None = None

        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps, train_config.vpp, vpp_chunk_id)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.share_embeddings_and_output_weights = bool(config.tie_word_embeddings)
        self.vision_model: nn.Module | None = None

        self.embed: VocabParallelEmbedding | None = None
        if layout.has_embed:
            self.embed = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)

        recompute_modules = getattr(train_config, "recompute_modules", [])
        moe_act_recompute = "moe_act" in recompute_modules and "moe" not in recompute_modules
        self.layers = nn.ModuleList(
            [
                KimiK2Layer(
                    config,
                    ps,
                    idx,
                    use_deepep=train_config.use_deepep,
                    router_bias_rate=router_bias_rate,
                    fp8=train_config.fp8,
                    moe_act_recompute=moe_act_recompute,
                    use_thd=use_thd,
                )
                for idx in self.layer_indices
            ]
        )

        self.norm: nn.Module | None = None
        self.head: VocabParallelOutput | None = None
        if layout.has_head:
            self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)

        self.sp_params: list[nn.Parameter] = []
        if ps.tp_size > 1:
            self.sp_params = _collect_sp_grad_params(self)

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            if len(input_tensor) > 1:
                raise ValueError("KimiK2Model expects a single pipeline input tensor.")
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        use_fused_kernels: bool = False,
        calculate_entropy: bool = False,
    ) -> dict:
        del position_ids, loss_mask
        if self.embed is not None:
            assert input_ids is not None
            h = self.embed(input_ids)
        else:
            if hidden_states is None:
                hidden_states = self._input_tensor
            assert hidden_states is not None
            h = hidden_states

        fp8_ctx = (
            te.fp8_autocast(enabled=True, fp8_recipe=build_fp8_recipe(self.train_config))
            if self.train_config.fp8
            else nullcontext()
        )
        with fp8_ctx:
            if self.embed is not None:
                h = scatter_to_sequence_parallel(h, self.ps)
            for layer in self.layers:
                h = layer(h, packed_seq_params=packed_seq_params)

        output = {"hidden_states": h}
        if self.head is not None:
            assert self.norm is not None
            hidden_for_head = self.norm(h)
            if labels is not None:
                temperature_value = _temperature_to_float(temperature)
                labels_sb = labels.transpose(0, 1).contiguous()
                if use_fused_kernels:
                    hidden_full = gather_from_sequence_parallel(hidden_for_head, self.ps)
                    log_probs, entropy = linear_cross_entropy(
                        hidden_full,
                        self._head_weight_for_fused_ce(hidden_full),
                        labels_sb,
                        temperature_value,
                        self.ps.tp_group,
                    )
                    output["loss"] = (-log_probs).mean()
                    output["log_probs"] = log_probs.transpose(0, 1).contiguous()
                    if calculate_entropy:
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
                else:
                    logits = self.head(hidden_for_head)
                    if temperature_value != 1.0:
                        logits = logits / temperature_value
                    loss = vocab_parallel_cross_entropy(logits, labels_sb, self.ps.tp_group)
                    output["loss"] = loss.mean()
                    output["log_probs"] = (-loss).transpose(0, 1).contiguous()
                    if calculate_entropy:
                        entropy = vocab_parallel_entropy(logits, self.ps.tp_group)
                        output["entropy"] = entropy.transpose(0, 1).contiguous()
            else:
                logits = self.head(hidden_for_head)
                output["logits"] = self.head.gather(logits).transpose(0, 1).contiguous()
        return output

    def _head_weight_for_fused_ce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.head is not None
        weight = self.head.col.linear.weight
        return weight if weight.dtype == hidden_states.dtype else weight.to(dtype=hidden_states.dtype)


__all__ = [
    "DenseMLP",
    "KimiK2Layer",
    "KimiK2Model",
    "MoELayer",
    "MultiLatentAttention",
    "SharedExpert",
]
