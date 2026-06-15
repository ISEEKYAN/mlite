# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.modules.attention.hca import HyperConnection
from megatron.lite.primitive.ops.logprob import vocab_parallel_entropy
from megatron.lite.primitive.modules.attention.mhc import MultiHeadHyperConnectionHead
from megatron.lite.primitive.parallel.mhc import (
    contract_mhc_hidden_for_pipeline,
    expand_mhc_hidden_for_pipeline,
)
from megatron.lite.primitive.parallel.pp import build_pipeline_chunk_layout
from megatron.lite.primitive.parallel.state import ParallelState


def _roll_mtp_sequence(tensor: torch.Tensor, *, seq_dim: int = 1) -> torch.Tensor:
    rolled = torch.roll(tensor, shifts=-1, dims=seq_dim)
    index = [slice(None)] * rolled.dim()
    index[seq_dim] = -1
    rolled[tuple(index)] = 0
    return rolled


class DeepseekV4Layer(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        ps: ParallelState,
        *,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.ps = ps
        self.self_attn = CompressedSparseAttention(
            config,
            layer_idx=layer_idx,
            ps=ps,
        )
        self.mlp = DeepseekV4MoE(config, ps, layer_idx=layer_idx, use_deepep=use_deepep)
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )
        self.ffn_hc = HyperConnection(
            config.hidden_size, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        attn_in, post, comb = self.attn_hc(x)
        attn_out = self.self_attn(
            self.input_layernorm(attn_in),
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        x = HyperConnection.post(attn_out, residual, post, comb)

        residual = x
        ffn_in, post, comb = self.ffn_hc(x)
        ffn_out = self.mlp(self.post_attention_layernorm(ffn_in), input_ids=input_ids)
        return HyperConnection.post(ffn_out, residual, post, comb)


class DeepseekV4MTPBlock(DeepseekV4Layer):
    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        ps: ParallelState,
        *,
        use_deepep: bool = False,
    ):
        super().__init__(
            config,
            layer_idx=layer_idx,
            ps=ps,
            use_deepep=use_deepep,
        )
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = MultiHeadHyperConnectionHead(
            config.hidden_size, config.hc_mult, config.hc_eps
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        embed_tokens: nn.Embedding,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedded = self.enorm(embed_tokens(input_ids))
        projected = self.e_proj(embedded).unsqueeze(2) + self.h_proj(self.hnorm(x))
        out = super().forward(
            projected,
            position_ids=position_ids,
            attention_mask=attention_mask,
            input_ids=input_ids,
        )
        return out

    def contract(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.hc_head(x))


class DeepseekV4Model(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        ps: ParallelState,
        *,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        layout = build_pipeline_chunk_layout(config.num_hidden_layers, ps, vpp, vpp_chunk_id)
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size) if layout.has_embed else None
        )
        self.layers = nn.ModuleDict(
            {
                str(i): DeepseekV4Layer(
                    config,
                    layer_idx=i,
                    ps=ps,
                    use_deepep=use_deepep,
                )
                for i in self.layer_indices
            }
        )
        self.norm = (
            te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if layout.has_head else None
        )
        self.hc_head = (
            MultiHeadHyperConnectionHead(config.hidden_size, config.hc_mult, config.hc_eps)
            if layout.has_head
            else None
        )
        self.mtp = (
            nn.ModuleList(
                [
                    DeepseekV4MTPBlock(
                        config,
                        layer_idx=config.num_hidden_layers + i,
                        ps=ps,
                        use_deepep=use_deepep,
                    )
                    for i in range(config.num_nextn_predict_layers)
                ]
            )
            if layout.has_head
            else nn.ModuleList()
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        return_mtp_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if self.embed_tokens is not None:
            if input_ids is None:
                raise ValueError("input_ids is required on the first DeepSeek V4 PP stage.")
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        elif hidden_states is None:
            raise ValueError("hidden_states is required on non-first DeepSeek V4 PP stages.")
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if self.embed_tokens is not None:
            hidden = expand_mhc_hidden_for_pipeline(
                self.embed_tokens(input_ids),
                hc_mult=self.config.hc_mult,
            )
            batch, seq_len = input_ids.shape
        else:
            hidden = hidden_states
            hidden = expand_mhc_hidden_for_pipeline(hidden, hc_mult=self.config.hc_mult)
            batch, seq_len = hidden.size(0), hidden.size(1)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else hidden.device
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        for layer in self.layers.values():
            hidden = layer(
                hidden,
                position_ids=position_ids,
                attention_mask=attention_mask,
                input_ids=input_ids,
            )
        return contract_mhc_hidden_for_pipeline(
            hidden,
            norm=self.norm,
            head=self.hc_head,
            return_source=return_mtp_source,
        )


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
        train_cfg: SimpleNamespace | None = None,
        ps: ParallelState | None = None,
        *,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        use_deepep: bool = False,
    ):
        super().__init__()
        self.config = config
        self.train_cfg = train_cfg or SimpleNamespace(fp8=False)
        self.ps = ps or ParallelState()
        self.model = DeepseekV4Model(
            config,
            self.ps,
            vpp=vpp,
            vpp_chunk_id=vpp_chunk_id,
            use_deepep=use_deepep,
        )
        self.pre_process, self.post_process = self.model.pre_process, self.model.post_process
        self.share_embeddings_and_output_weights = False
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if self.model.post_process
            else None
        )
        self._input_tensor: torch.Tensor | None = None

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0] if input_tensor else None
        self._input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
        enable_mtp: bool = True,
    ) -> dict[str, Any]:
        if hidden_states is None:
            hidden_states = self._input_tensor
        if input_ids is None and hidden_states is None:
            raise ValueError("input_ids or hidden_states is required")
        run_mtp = (
            input_ids is not None
            and self.model.post_process
            and self.lm_head is not None
            and len(self.model.mtp) > 0
            and self.ps.cp_size == 1
            and enable_mtp
            and self.model.embed_tokens is not None
        )
        fp8_ctx = nullcontext()
        with fp8_ctx:
            model_out = self.model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                return_mtp_source=run_mtp,
            )
            if run_mtp:
                hidden, mtp_source = model_out
                assert mtp_source is not None
            else:
                hidden = model_out
                mtp_source = None
            logits = None if self.lm_head is None else self.lm_head(hidden)
        output = {"hidden_states": hidden}
        if logits is None:
            return output
        if isinstance(temperature, torch.Tensor):
            temperature = float(temperature.detach().float().item())
        if temperature != 1.0:
            logits = logits / float(temperature)
        output["logits"] = logits
        if run_mtp:
            assert input_ids is not None
            if position_ids is None:
                position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            mtp_input_ids = input_ids
            mtp_hidden_states = []
            mtp_logits = []
            for mtp_layer in self.model.mtp:
                mtp_input_ids = _roll_mtp_sequence(mtp_input_ids)
                mtp_source = mtp_layer(
                    mtp_source,
                    input_ids=mtp_input_ids,
                    embed_tokens=self.model.embed_tokens,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )
                mtp_hidden = mtp_layer.contract(mtp_source)
                mtp_logit = self.lm_head(mtp_hidden)
                if temperature != 1.0:
                    mtp_logit = mtp_logit / float(temperature)
                mtp_hidden_states.append(mtp_hidden)
                mtp_logits.append(mtp_logit)
            output["mtp_hidden_states"] = tuple(mtp_hidden_states)
            output["mtp_logits"] = tuple(mtp_logits)
        if labels is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            if loss_mask is not None:
                valid = loss_mask.reshape(-1).float()
                loss = (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss = token_loss.mean()
            output["loss"] = loss
            output["log_probs"] = (-token_loss).view_as(labels).contiguous()
            if calculate_entropy:
                output["entropy"] = vocab_parallel_entropy(logits, self.ps.tp_group).contiguous()
            if run_mtp and self.config.num_nextn_predict_layers > 0:
                mtp_labels = labels
                mtp_loss_mask = (
                    torch.ones_like(labels, dtype=torch.float32)
                    if loss_mask is None
                    else loss_mask.float()
                )
                mtp_losses = []
                for mtp_logit in output["mtp_logits"]:
                    mtp_labels = _roll_mtp_sequence(mtp_labels)
                    mtp_loss_mask = _roll_mtp_sequence(mtp_loss_mask)
                    mtp_token_loss = F.cross_entropy(
                        mtp_logit.reshape(-1, mtp_logit.size(-1)).float(),
                        mtp_labels.reshape(-1),
                        ignore_index=-100,
                        reduction="none",
                    ).view_as(mtp_labels)
                    valid = mtp_loss_mask.reshape(-1).float()
                    mtp_losses.append(
                        (mtp_token_loss.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0)
                    )
                if mtp_losses:
                    output["mtp_loss"] = torch.stack(mtp_losses).mean()
                    output["loss"] = (
                        output["loss"] + self.config.mtp_loss_scaling_factor * output["mtp_loss"]
                    )
        return output
