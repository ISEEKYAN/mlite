# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native GLM-5 (deepseek_v3_2) model.

Structurally a Kimi-K2 / DeepSeek-V3 decoder that reuses the SAME shared MoE
primitives (``Experts``, sigmoid-topk router with aux load-balancing loss,
``TokenDispatcher``, ``MoEAuxLossAutoScaler``).  The only architectural
difference from Kimi is the attention module: GLM-5 swaps MLA for the
Dynamic Sparse Attention (DSA) primitive ``DynamicSparseAttention`` (+ indexer).

DSA is a model-agnostic primitive operating on batch-first ``[B, S, H]`` tensors
with explicit ``cos``/``sin``/``position_ids`` and plain (non-TP) linears, so the
model keeps the DeepSeek-V4 family's batch-first layout rather than Kimi's
sequence-parallel SBHD layout (see report for the SBHD/BSHD rationale).  Expert
compute, routing and dispatch are nonetheless the shared Kimi primitives.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.primitive.modules.attention import (
    DynamicSparseAttention,
    RMSNorm,
    build_rotary_embeddings,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.mlp import SwiGLUMLP
from megatron.lite.primitive.modules.moe import MoEAuxLossAutoScaler
from megatron.lite.primitive.modules.router import (
    SigmoidTopKRouter,
    _ordered_topk_from_routing_map,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.pp import build_pipeline_chunk_layout
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from megatron.lite.primitive.utils.moe import (
    compute_routing_scores_for_aux_loss,
    switch_load_balancing_loss_func,
    topk_routing_with_score_function,
)


class Glm5SigmoidTopKRouter(SigmoidTopKRouter):
    """GLM-5 sigmoid router.

    Structurally the shared ``SigmoidTopKRouter`` (sigmoid scoring + expert-bias
    bias-correction + per-token norm + ``routed_scaling_factor`` + aux
    load-balancing loss via ``MoEAuxLossAutoScaler`` + tp-gated all-reduce of
    token counts).  Mirrors Kimi's ``KimiK2SigmoidTopKRouter`` but additionally
    threads GLM-5's group-limited routing config (``n_group``/``topk_group``)
    into the shared ``topk_routing_with_score_function``.

    GLM-5's config drives ``num_experts_per_tok`` / ``routed_scaling_factor`` /
    ``n_group`` / ``topk_group``.  GLM-5 has no ``scoring_func`` /
    ``aux_loss_alpha`` HF fields, so sigmoid scoring and a zero aux coefficient
    are used by default (aux loss therefore contributes 0, matching the old
    no-aux behaviour, while keeping the kimi structure available).
    """

    def __init__(self, config: Glm5Config, ps: ParallelState, *, compute_aux_loss: bool = True):
        # The shared router reads num_experts_per_tok / n_routed_experts /
        # routed_scaling_factor / scoring_func / aux_loss_alpha from the config.
        super().__init__(config, ps, compute_aux_loss=compute_aux_loss)
        # GLM-5 ships a persistent sigmoid bias-correction term
        # (HF `mlp.gate.e_score_correction_bias`).  The shared router registers
        # `expert_bias` as non-persistent; make it persistent so it round-trips
        # through load_hf / save_hf.
        self._non_persistent_buffers_set.discard("expert_bias")
        n_group = getattr(config, "n_group", None)
        topk_group = getattr(config, "topk_group", None)
        # Group-limited routing only when groups are actually used (>1).
        self.num_groups = n_group if (n_group and n_group > 1) else None
        self.group_topk = topk_group if self.num_groups is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        logits = logits.view(-1, self.num_experts)
        num_tokens = logits.size(0)
        routing_kwargs: dict = {}
        if self.num_groups is not None and self.group_topk is not None:
            routing_kwargs = dict(num_groups=self.num_groups, group_topk=self.group_topk)
        probs_dense, routing_map = topk_routing_with_score_function(
            logits,
            self.topk,
            score_function=self.score_function,
            expert_bias=self.expert_bias.to(logits.dtype),
            scaling_factor=(self.scaling_factor or None),
            fused=self.moe_router_fusion,
            **routing_kwargs,
        )
        topk_scores, topk_indices = _ordered_topk_from_routing_map(
            probs_dense, routing_map, self.topk
        )
        topk_scores = topk_scores.to(logits.dtype)

        if self.compute_aux_loss and self.training and torch.is_grad_enabled():
            _, aux_scores = compute_routing_scores_for_aux_loss(
                logits, self.topk, score_function=self.score_function, fused=self.moe_router_fusion
            )
            tokens_per_expert = routing_map.sum(dim=0).to(torch.int64)
            total_num_tokens = num_tokens
            if self._aux_loss_group is not None:
                dist.all_reduce(tokens_per_expert, group=self._aux_loss_group)
                total_num_tokens = num_tokens * dist.get_world_size(group=self._aux_loss_group)
            aux_loss = switch_load_balancing_loss_func(
                aux_scores,
                tokens_per_expert,
                total_num_tokens,
                self.topk,
                self.num_experts,
                self.aux_loss_coeff,
                fused=False,
            )
            topk_scores = MoEAuxLossAutoScaler.apply(topk_scores, aux_loss)

        return topk_scores, topk_indices


class Glm5MoE(nn.Module):
    """MoE assembly mirroring Kimi's MoELayer: router -> dispatcher -> shared
    Experts -> combine, plus a shared expert.

    Reuses the shared ``Experts`` / ``TokenDispatcher`` / ``SigmoidTopKRouter``
    primitives (no hand-rolled per-expert Python loop, no reinvented router).
    """

    def __init__(
        self, config: Glm5Config, ps: ParallelState, *, use_deepep: bool = False
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.gate = Glm5SigmoidTopKRouter(config, ps, compute_aux_loss=True)
        self.experts = Experts(config, ps)
        self.dispatcher = TokenDispatcher(
            config.n_routed_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
        )
        shared_intermediate = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = (
            SwiGLUMLP(config.hidden_size, shared_intermediate)
            if config.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, self.hidden_size)
        scores, indices = self.gate(x_flat)
        dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x_flat, scores, indices)
        del scores, indices
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            tpe,
            permuted_probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        out = self.dispatcher.combine(expert_out)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x_flat)
        return out.view(shape).to(x.dtype)


class Glm5MLP(SwiGLUMLP):
    """Dense MLP, reusing the shared SwiGLU primitive.

    ``gate_up`` is the fused gate+up projection (HF ``gate_proj``/``up_proj`` are
    concatenated by the checkpoint loader / split on export); ``down`` is the
    output projection.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__(hidden_size, intermediate_size)


class Glm5Layer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        layer_idx: int,
        ps: ParallelState | None = None,
        *,
        use_deepep: bool = False,
    ):
        super().__init__()
        ps = ps or ParallelState()
        self.layer_idx = layer_idx
        self.self_attn = DynamicSparseAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            rope_interleaved=config.rope_interleave,
            latent_rms_norm_eps=config.latent_rms_norm_eps,
            indexer_layer_norm_eps=config.indexer_layer_norm_eps,
            indexer_rope_interleaved=config.indexer_rope_interleave,
            indexer_rope_first=config.indexer_rope_first,
            indexer_use_hadamard=config.indexer_use_hadamard,
            indexer_loss_coeff=config.dsa_indexer_loss_coeff,
            indexer_use_sparse_loss=config.dsa_indexer_use_sparse_loss,
            calculate_per_token_loss=config.calculate_per_token_loss,
            cp_size=ps.cp_size,
            cp_rank=ps.cp_rank,
            cp_group=ps.cp_group,
        )
        if config.is_moe_layer(layer_idx):
            self.mlp: nn.Module = Glm5MoE(config, ps, use_deepep=use_deepep)
        else:
            self.mlp = Glm5MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        attn_out = self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def _roll_mtp_sequence(
    tensor: torch.Tensor, *, seq_dim: int = 1, packed_seq_params=None
) -> torch.Tensor:
    if packed_seq_params is not None:
        rolled, _ = roll_packed_thd_left(tensor, packed_seq_params=packed_seq_params, dims=seq_dim)
        return rolled
    rolled = torch.roll(tensor, shifts=-1, dims=seq_dim)
    index = [slice(None)] * rolled.dim()
    index[seq_dim] = -1
    rolled[tuple(index)] = 0
    return rolled


class Glm5MTPLayer(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        layer_idx: int,
        ps: ParallelState,
        *,
        embedding: nn.Embedding,
        use_deepep: bool = False,
        detach_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        object.__setattr__(self, "embedding", embedding)
        self.detach_encoder = detach_encoder
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.transformer_layer = Glm5Layer(config, layer_idx, ps, use_deepep=use_deepep)
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotary_position_ids = position_ids
        input_ids = _roll_mtp_sequence(input_ids, packed_seq_params=packed_seq_params)
        if packed_seq_params is None or position_ids.shape[-1] == input_ids.shape[-1]:
            position_ids = _roll_mtp_sequence(position_ids, packed_seq_params=packed_seq_params)
        decoder_input = self.embedding(input_ids)
        if self.detach_encoder:
            decoder_input = decoder_input.detach()
            hidden_states = hidden_states.detach()
        decoder_input = self.enorm(decoder_input)
        hidden_states = self.hnorm(hidden_states)
        hidden_states = self.eh_proj(torch.cat((decoder_input, hidden_states), dim=-1))
        cos, sin = build_rotary_embeddings(
            position_ids=rotary_position_ids.to(device=hidden_states.device, dtype=torch.long),
            dim=self.config.qk_rope_head_dim,
            rope_theta=self.config.rope_theta,
            dtype=hidden_states.dtype,
        )
        hidden_states = self.transformer_layer(
            hidden_states,
            cos=cos,
            sin=sin,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, input_ids, position_ids


class Glm5MTPBlock(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState,
        *,
        embedding: nn.Embedding,
        use_deepep: bool = False,
        detach_encoder: bool = False,
        repeated_layer: bool = False,
    ):
        super().__init__()
        self.num_layers = config.num_nextn_predict_layers
        self.repeated_layer = bool(repeated_layer)
        layers_to_build = 1 if self.repeated_layer else self.num_layers
        self.layers = nn.ModuleList(
            [
                Glm5MTPLayer(
                    config,
                    config.num_hidden_layers + layer_idx,
                    ps,
                    embedding=embedding,
                    use_deepep=use_deepep,
                    detach_encoder=detach_encoder,
                )
                for layer_idx in range(layers_to_build)
            ]
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for depth in range(self.num_layers):
            layer = self.layers[0] if self.repeated_layer else self.layers[depth]
            hidden_states, input_ids, position_ids = layer(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
            outputs.append(hidden_states)
        return outputs


class Glm5Model(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        ps: ParallelState | None = None,
        *,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        use_deepep: bool = False,
        mtp_enable: bool = False,
        mtp_detach_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        layout = build_pipeline_chunk_layout(
            config.num_hidden_layers, self.ps, vpp, vpp_chunk_id
        )
        self.layer_indices = layout.layer_indices
        self.pre_process = layout.has_embed
        self.post_process = layout.has_head
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size) if layout.has_embed else None
        )
        self.layers = nn.ModuleList(
            [Glm5Layer(config, i, self.ps, use_deepep=use_deepep) for i in self.layer_indices]
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if layout.has_head else None
        )
        self.mtp_embed: nn.Embedding | None = None
        self.mtp: Glm5MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and layout.has_head:
            mtp_embedding = self.embed_tokens
            if mtp_embedding is None:
                mtp_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
                self.mtp_embed = mtp_embedding
            self.mtp = Glm5MTPBlock(
                config,
                self.ps,
                embedding=mtp_embedding,
                use_deepep=use_deepep,
                detach_encoder=mtp_detach_encoder,
                repeated_layer=config.mtp_use_repeated_layer,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        packed_seq_params=None,
    ) -> torch.Tensor:
        if hidden_states is None:
            if input_ids is None:
                raise ValueError("input_ids or hidden_states is required")
            if self.embed_tokens is None:
                raise ValueError("input_ids are only accepted on the first pipeline stage")
            hidden_states = self.embed_tokens(input_ids)
        elif self.embed_tokens is None:
            # Received from the pipeline P2P, which uses the Megatron SBHD
            # [S, B, H] convention; the DSA layers run batch-first [B, S, H].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        batch, seq_len, _ = hidden_states.shape
        if position_ids is None and packed_seq_params is not None:
            raise ValueError("GLM5 packed THD forward requires explicit position_ids.")
        if position_ids is None:
            if self.ps.cp_size > 1:
                full_seq_len = seq_len * self.ps.cp_size
                position_ids = (
                    torch.arange(full_seq_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )
            else:
                position_ids = (
                    torch.arange(seq_len, device=hidden_states.device)
                    .unsqueeze(0)
                    .expand(batch, -1)
                )
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)

        cos, sin = build_rotary_embeddings(
            position_ids=position_ids.to(device=hidden_states.device, dtype=torch.long),
            dim=self.config.qk_rope_head_dim,
            rope_theta=self.config.rope_theta,
            dtype=hidden_states.dtype,
        )

        h = hidden_states
        for layer in self.layers:
            h = layer(
                h,
                cos=cos,
                sin=sin,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
        if self.norm is not None:
            return self.norm(h)
        # Non-last stage: hand back to the pipeline P2P in SBHD [S, B, H].
        return h.transpose(0, 1).contiguous()


class Glm5ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        train_cfg: SimpleNamespace | None = None,
        ps: ParallelState | None = None,
        *,
        vpp: int | None = None,
        vpp_chunk_id: int | None = None,
        mtp_enable: bool = False,
        mtp_enable_train: bool = False,
        mtp_detach_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.train_cfg = train_cfg or SimpleNamespace(fp8=False)
        self.ps = ps or ParallelState()
        self.mtp_enable_train = bool(mtp_enable and mtp_enable_train)
        self.model = Glm5Model(
            config,
            self.ps,
            vpp=vpp,
            vpp_chunk_id=vpp_chunk_id,
            use_deepep=bool(getattr(self.train_cfg, "use_deepep", False)),
            mtp_enable=mtp_enable,
            mtp_detach_encoder=mtp_detach_encoder,
        )
        self.layer_indices = self.model.layer_indices
        self.pre_process = self.model.pre_process
        self.post_process = self.model.post_process
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
        packed_seq_params=None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        temperature: float | torch.Tensor = 1.0,
        calculate_entropy: bool = False,
    ) -> dict[str, torch.Tensor]:
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if hidden_states is None:
            hidden_states = self._input_tensor

        fp8_ctx = nullcontext()
        with fp8_ctx:
            hidden = self.model(
                input_ids,
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
            logits = self.lm_head(hidden) if self.lm_head is not None else None

        if isinstance(temperature, torch.Tensor):
            temperature = float(temperature.detach().float().item())
        if logits is not None and temperature != 1.0:
            logits = logits / float(temperature)

        output = {"hidden_states": hidden}
        if logits is None:
            return output
        output["logits"] = logits

        mtp_hidden_states = self._apply_mtp(
            hidden,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )
        if mtp_hidden_states is not None:
            output["mtp_hidden_states"] = tuple(mtp_hidden_states)
            mtp_logits = tuple(
                (
                    self.lm_head(mtp_hidden) / float(temperature)
                    if temperature != 1.0
                    else self.lm_head(mtp_hidden)
                )
                for mtp_hidden in mtp_hidden_states
            )
            output["mtp_logits"] = mtp_logits

        if labels is not None:
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            output["log_probs"] = (-token_loss).view_as(labels).contiguous()
            if loss_mask is not None:
                valid = loss_mask.reshape(-1).float()
                loss = (token_loss * valid).sum() / valid.sum().clamp_min(1.0)
            else:
                loss = token_loss.mean()
            output["loss"] = loss
            if calculate_entropy:
                probs = torch.softmax(logits.float(), dim=-1)
                output["entropy"] = -(probs * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
            mtp_loss = self._apply_mtp_loss(
                output.get("mtp_logits"),
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )
            if mtp_loss is not None:
                output["mtp_loss"] = mtp_loss
                output["loss"] = output["loss"] + self.config.mtp_loss_scaling_factor * mtp_loss
        else:
            if calculate_entropy:
                probs = torch.softmax(logits.float(), dim=-1)
                output["entropy"] = -(probs * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
        return output

    def _apply_mtp(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        packed_seq_params,
    ) -> list[torch.Tensor] | None:
        if self.model.mtp is None:
            return None
        if input_ids is None:
            if self.mtp_enable_train:
                raise ValueError("MTP training requires input_ids.")
            return None
        batch, seq_len = input_ids.shape
        if position_ids is None:
            if packed_seq_params is not None:
                raise ValueError("GLM5 MTP packed THD forward requires explicit position_ids.")
            position_ids = (
                torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(batch, -1)
            )
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch, -1)
        return self.model.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            packed_seq_params=packed_seq_params,
        )

    def _apply_mtp_loss(
        self,
        mtp_logits: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor | None,
        *,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
        packed_seq_params=None,
    ) -> torch.Tensor | None:
        if mtp_logits is None or not self.mtp_enable_train:
            return None
        logits_list = [mtp_logits] if isinstance(mtp_logits, torch.Tensor) else list(mtp_logits)
        if loss_mask is None:
            mtp_loss_mask = torch.ones_like(labels, dtype=torch.float32)
        else:
            mtp_loss_mask = loss_mask.to(dtype=torch.float32).clone()
        mtp_labels = labels.clone()
        losses = []
        for logits in logits_list:
            mtp_labels = _roll_mtp_sequence(mtp_labels, packed_seq_params=packed_seq_params)
            mtp_loss_mask = _roll_mtp_sequence(mtp_loss_mask, packed_seq_params=packed_seq_params)
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                mtp_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(mtp_labels)
            valid = mtp_loss_mask.reshape(-1).float()
            losses.append((token_loss.reshape(-1) * valid).sum() / valid.sum().clamp_min(1.0))
        if not losses:
            return None
        return torch.stack(losses).mean()

    @torch.no_grad()
    def initialize_weights(self) -> None:
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, Experts):
                for param in module.parameters():
                    nn.init.normal_(param, mean=0.0, std=std)
            elif isinstance(module, Glm5SigmoidTopKRouter):
                nn.init.normal_(module.gate.weight, mean=0.0, std=std)
                if module.expert_bias is not None:
                    module.expert_bias.zero_()
            elif hasattr(module, "reset_parameters") and isinstance(module, RMSNorm):
                module.reset_parameters()


__all__ = [
    "Glm5ForCausalLM",
    "Glm5Layer",
    "Glm5MLP",
    "Glm5MTPBlock",
    "Glm5MTPLayer",
    "Glm5MoE",
    "Glm5Model",
    "Glm5SigmoidTopKRouter",
]
