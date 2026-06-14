"""Native GLM-5 model assembled from Megatron Lite primitives."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.primitive.attention import (
    DynamicSparseAttention,
    RMSNorm,
    build_rotary_embeddings,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.parallel.thd import roll_packed_thd_left
from megatron.lite.primitive.utils import ensure_divisible


class Glm5MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Glm5RoutedExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        ps: ParallelState | None = None,
    ):
        super().__init__()
        ps = ps or ParallelState()
        self.num_global_experts = num_experts
        self.num_experts = ensure_divisible(num_experts, ps.ep_size)
        self.local_start = ps.ep_rank * self.num_experts
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, hidden_size, intermediate_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate_up_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.size(0) == 0:
            output = hidden_states.reshape(0, self.hidden_size)
            if permuted_probs is not None:
                output = output + permuted_probs.sum().to(output.dtype) * 0.0
            return output

        output = hidden_states.new_empty(hidden_states.size(0), self.hidden_size)
        offset = 0
        for expert_idx, count in enumerate(tokens_per_expert.tolist()):
            if count == 0:
                continue
            end = offset + count
            expert_input = hidden_states[offset:end]
            gate_up = F.linear(expert_input, self.gate_up_proj[expert_idx])
            gate_proj, up_proj = gate_up.chunk(2, dim=-1)
            expert_hidden = F.silu(gate_proj) * up_proj
            expert_out = F.linear(expert_hidden, self.down_proj[expert_idx])
            if permuted_probs is not None:
                expert_out = expert_out * permuted_probs[offset:end].unsqueeze(-1)
            output[offset:end] = expert_out.to(hidden_states.dtype)
            offset = end
        return output

    def forward_topk(
        self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        num_topk = topk_indices.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)

        token_idx = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, num_topk)
            .reshape(-1)
        )
        sample_weights = topk_weights.reshape(-1)
        expert_ids = topk_indices.reshape(-1)
        invalid_mask = expert_ids >= self.num_experts
        expert_ids = expert_ids.clamp(0, self.num_experts - 1)
        selected_hidden = hidden_states[token_idx]

        selected_gate_up = self.gate_up_proj[expert_ids]
        gate_up = torch.bmm(selected_gate_up, selected_hidden.unsqueeze(-1)).squeeze(-1)
        gate_proj, up_proj = gate_up.chunk(2, dim=-1)
        expert_hidden = F.silu(gate_proj) * up_proj

        selected_down = self.down_proj[expert_ids]
        expert_out = torch.bmm(selected_down, expert_hidden.unsqueeze(-1)).squeeze(-1)
        weighted = expert_out * sample_weights.unsqueeze(-1)
        weighted.masked_fill_(invalid_mask.unsqueeze(-1), 0.0)
        return weighted.view(num_tokens, num_topk, hidden_dim).sum(dim=1).to(hidden_states.dtype)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        for expert_idx in range(self.num_experts):
            gate_proj = self.gate_up_proj[expert_idx, : self.intermediate_size]
            up_proj = self.gate_up_proj[expert_idx, self.intermediate_size :]
            down_proj = self.down_proj[expert_idx]
            if not keep_vars:
                gate_proj = gate_proj.detach()
                up_proj = up_proj.detach()
                down_proj = down_proj.detach()
            expert_prefix = f"{prefix}{expert_idx}."
            destination[expert_prefix + "gate_proj.weight"] = gate_proj
            destination[expert_prefix + "up_proj.weight"] = up_proj
            destination[expert_prefix + "down_proj.weight"] = down_proj

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ) -> None:
        del local_metadata

        def _copy(key: str, target: torch.Tensor) -> bool:
            tensor = state_dict.get(key)
            if tensor is None:
                if strict:
                    missing_keys.append(key)
                return False
            if tensor.shape != target.shape:
                error_msgs.append(
                    f"size mismatch for {key}: copying a param with shape {tuple(tensor.shape)} "
                    f"from checkpoint, the shape in current model is {tuple(target.shape)}."
                )
                return False
            with torch.no_grad():
                target.copy_(tensor)
            return True

        gate_up_key = prefix + "gate_up_proj"
        if gate_up_key in state_dict:
            _copy(gate_up_key, self.gate_up_proj)
        else:
            for expert_idx in range(self.num_experts):
                expert_prefix = f"{prefix}{expert_idx}."
                _copy(
                    expert_prefix + "gate_proj.weight",
                    self.gate_up_proj[expert_idx, : self.intermediate_size],
                )
                _copy(
                    expert_prefix + "up_proj.weight",
                    self.gate_up_proj[expert_idx, self.intermediate_size :],
                )
        down_key = prefix + "down_proj"
        if down_key in state_dict:
            _copy(down_key, self.down_proj)
        else:
            for expert_idx in range(self.num_experts):
                _copy(f"{prefix}{expert_idx}.down_proj.weight", self.down_proj[expert_idx])


class Glm5Router(nn.Module):
    def __init__(self, config: Glm5Config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.register_buffer(
            "e_score_correction_bias", torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.topk = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.route_scale = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float()).sigmoid()
        original_scores = scores
        scores_for_choice = scores + self.e_score_correction_bias.to(scores.dtype)

        if self.n_group > 1:
            grouped = scores_for_choice.view(x.size(0), self.n_group, -1)
            group_scores = grouped.topk(2, dim=-1, sorted=False).values.sum(dim=-1)
            group_idx = group_scores.topk(self.topk_group, dim=-1, sorted=False).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
                1, group_idx, True
            )
            scores_for_choice = scores_for_choice.masked_fill(
                ~group_mask.unsqueeze(-1).expand_as(grouped).flatten(1), 0.0
            )

        indices = scores_for_choice.topk(self.topk, dim=-1, sorted=False).indices
        weights = original_scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return weights * self.route_scale, indices


def _build_glm5_pipeline_layers(num_hidden_layers: int, ps: ParallelState) -> list[int]:
    if ps.pp_size > 2 and num_hidden_layers % ps.pp_size:
        middle_layers = -(-num_hidden_layers // ps.pp_size)
        edge_layers = num_hidden_layers - middle_layers * (ps.pp_size - 2)
        first_layers = edge_layers // 2
        last_layers = edge_layers - first_layers
        counts = [first_layers, *([middle_layers] * (ps.pp_size - 2)), last_layers]
        start = sum(counts[: ps.pp_rank])
        return list(range(start, start + counts[ps.pp_rank]))

    layers_per_stage = num_hidden_layers // ps.pp_size
    remainder = num_hidden_layers % ps.pp_size
    local_count = layers_per_stage + (1 if ps.pp_rank < remainder else 0)
    start = ps.pp_rank * layers_per_stage + min(ps.pp_rank, remainder)
    return list(range(start, start + local_count))


class Glm5MoE(nn.Module):
    def __init__(
        self, config: Glm5Config, ps: ParallelState | None = None, *, use_deepep: bool = False
    ):
        super().__init__()
        ps = ps or ParallelState()
        self.hidden_size = config.hidden_size
        self.gate = Glm5Router(config)
        self.dispatcher = None
        if ps.ep_size > 1:
            from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

            self.dispatcher = TokenDispatcher(
                config.n_routed_experts,
                config.hidden_size,
                ps,
                use_deepep=use_deepep,
                fuse_score_alltoall=True,
            )
        self.experts = Glm5RoutedExperts(
            config.n_routed_experts, config.hidden_size, config.moe_intermediate_size, ps
        )
        shared_intermediate = config.n_shared_experts * config.moe_intermediate_size
        self.shared_experts = (
            Glm5MLP(config.hidden_size, shared_intermediate)
            if config.n_shared_experts > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        residual = x
        x_flat = x.reshape(-1, self.hidden_size)
        weights, indices = self.gate(x_flat)
        if self.dispatcher is None:
            y = self.experts.forward_topk(x_flat, indices, weights).view(shape)
        else:
            dispatched, tpe, permuted_probs = self.dispatcher.dispatch(x_flat, weights, indices)
            self.dispatcher.wait_dispatch_event()
            expert_out = self.experts(dispatched, tpe, permuted_probs)
            y = self.dispatcher.combine(expert_out).view(shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(residual)
        return y


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
        self.mlp = (
            Glm5MoE(config, ps, use_deepep=use_deepep)
            if config.is_moe_layer(layer_idx)
            else Glm5MLP(config.hidden_size, config.intermediate_size)
        )
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
        use_deepep: bool = False,
        mtp_enable: bool = False,
        mtp_detach_encoder: bool = False,
    ):
        super().__init__()
        self.config = config
        self.ps = ps or ParallelState()
        self.layer_indices = _build_glm5_pipeline_layers(config.num_hidden_layers, self.ps)
        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size) if self.ps.pp_is_first else None
        )
        self.layers = nn.ModuleList(
            [Glm5Layer(config, i, self.ps, use_deepep=use_deepep) for i in self.layer_indices]
        )
        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if self.ps.pp_is_last else None
        )
        self.mtp_embed: nn.Embedding | None = None
        self.mtp: Glm5MTPBlock | None = None
        if mtp_enable and config.num_nextn_predict_layers > 0 and self.ps.pp_is_last:
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
        return self.norm(h) if self.norm is not None else h


class Glm5ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Glm5Config,
        train_cfg: SimpleNamespace | None = None,
        ps: ParallelState | None = None,
        *,
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
            use_deepep=bool(getattr(self.train_cfg, "use_deepep", False)),
            mtp_enable=mtp_enable,
            mtp_detach_encoder=mtp_detach_encoder,
        )
        self.layer_indices = self.model.layer_indices
        self.lm_head = (
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            if self.ps.pp_is_last
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, RMSNorm | nn.LayerNorm):
                (
                    module.reset_parameters()
                    if hasattr(module, "reset_parameters")
                    else module.weight.fill_(1.0)
                )
            elif isinstance(module, Glm5Router):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                module.e_score_correction_bias.zero_()
            elif isinstance(module, Glm5RoutedExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
                nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)


__all__ = [
    "Glm5ForCausalLM",
    "Glm5Layer",
    "Glm5MLP",
    "Glm5MTPBlock",
    "Glm5MTPLayer",
    "Glm5MoE",
    "Glm5Model",
    "Glm5Router",
    "Glm5RoutedExperts",
]
