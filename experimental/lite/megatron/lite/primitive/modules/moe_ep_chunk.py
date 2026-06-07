"""Chunked EP MoE primitive.

This is the conservative Megatron Lite landing of Bumblebee's EP chunk path:
policy and per-chunk DeepEP forward are available as a standalone primitive,
while gradient-enabled calls keep using the ordinary dispatcher path until the
full-recompute fused backward is ported.
"""

from __future__ import annotations

from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.moe_ep_chunk_policy import (
    ChunkSpec,
    ep_chunk_ranges,
    parse_ep_chunk_spec,
    resolve_ep_chunk_count,
)
from megatron.lite.primitive.modules.dispatcher import TokenDispatcher
from megatron.lite.primitive.modules.experts import Experts
from megatron.lite.primitive.modules.lora import LoraConfig
from megatron.lite.primitive.modules.router import TopKRouter
from megatron.lite.primitive.parallel import ParallelState


class EPChunkedMoELayer(nn.Module):
    """MoE layer with optional token-row chunked DeepEP forward.

    Chunked dispatch/combine is used only when autograd is disabled.  Under
    autograd, the layer deliberately falls back to the regular dispatcher path,
    because DeepEP submit/finish calls are not sufficient to model the fused
    backward path that Bumblebee uses for training.
    """

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        num_chunks_ep_a2a_overlap: ChunkSpec | str = "auto",
        use_deepep: bool = True,
        router_bias_rate: float = 0.0,
        compute_aux_loss: bool = False,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
        dispatcher_buffer_prefix: str = "ep_chunk",
    ):
        super().__init__()
        self.config = config
        self.ps = ps
        self.chunk_spec = parse_ep_chunk_spec(num_chunks_ep_a2a_overlap)
        self.router = TopKRouter(
            config,
            ps,
            router_bias_rate=router_bias_rate,
            compute_aux_loss=compute_aux_loss,
        )
        self.experts = Experts(
            config,
            ps,
            fp8=fp8,
            moe_act_recompute=moe_act_recompute,
            lora_config=lora_config,
        )
        self.dispatcher = TokenDispatcher(
            config.num_experts,
            config.hidden_size,
            ps,
            use_deepep=use_deepep,
            buffer_slot=(dispatcher_buffer_prefix, "full"),
        )
        self._dispatcher_buffer_prefix = dispatcher_buffer_prefix
        self._chunk_dispatchers: list[TokenDispatcher] = [self.dispatcher]

        if use_deepep and ps.ep_size > 1 and not self.dispatcher.use_deepep:
            raise RuntimeError("EPChunkedMoELayer requires DeepEP when ep_size > 1.")

    def num_chunks(self, num_tokens: int) -> int:
        return resolve_ep_chunk_count(
            num_tokens,
            ep_size=self.ps.ep_size,
            hidden_size=self.config.hidden_size,
            spec=self.chunk_spec,
            direction="forward",
        )

    def chunk_ranges(self, num_tokens: int) -> list[tuple[int, int]]:
        return ep_chunk_ranges(
            num_tokens,
            self.num_chunks(num_tokens),
            weights_env=(
                "MEGATRON_LITE_EP_CHUNK_FWD_WEIGHTS",
                "MEGATRON_LITE_EP_CHUNK_WEIGHTS",
                "BUMBLEBEE_EP_CHUNK_FWD_WEIGHTS",
                "BUMBLEBEE_EP_CHUNK_WEIGHTS",
            ),
        )

    def _chunk_dispatcher(self, idx: int) -> TokenDispatcher:
        while len(self._chunk_dispatchers) <= idx:
            self._chunk_dispatchers.append(
                TokenDispatcher(
                    self.config.num_experts,
                    self.config.hidden_size,
                    self.ps,
                    use_deepep=True,
                    buffer_slot=(self._dispatcher_buffer_prefix, idx),
                )
            )
        return self._chunk_dispatchers[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.reshape(-1, x.size(-1)) if x.dim() == 3 else x
        ranges = self.chunk_ranges(x_2d.size(0))
        if (
            len(ranges) <= 1
            or torch.is_grad_enabled()
            or not self.dispatcher.use_deepep
        ):
            return self._forward_full(x_2d).view(input_shape).to(x.dtype)
        return self._forward_chunked_no_grad(x_2d, ranges).view(input_shape).to(x.dtype)

    def _forward_full(self, x_2d: torch.Tensor) -> torch.Tensor:
        scores, indices = self.router(x_2d)
        dispatched, local_tpe, probs = self.dispatcher.dispatch(x_2d, scores, indices)
        self.dispatcher.wait_dispatch_event()
        expert_out = self.experts(
            dispatched,
            local_tpe,
            probs,
            tokens_per_expert_list=getattr(self.dispatcher, "_local_tpe_list", None),
        )
        return self.dispatcher.combine(expert_out)

    def _forward_chunked_no_grad(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
    ) -> torch.Tensor:
        outputs = []
        for idx, (start, end) in enumerate(ranges):
            dispatcher = self._chunk_dispatcher(idx)
            x_chunk = x_2d[start:end]
            scores, indices = self.router(x_chunk)
            dispatch_state = dispatcher.submit_deepep_dispatch(
                x_chunk,
                scores,
                indices,
                async_finish=True,
            )
            dispatched, local_tpe, probs = dispatcher.finish_deepep_dispatch(
                dispatch_state
            )
            expert_out = self.experts(
                dispatched,
                local_tpe,
                probs,
                tokens_per_expert_list=getattr(dispatcher, "_local_tpe_list", None),
            )
            combine_state = dispatcher.submit_deepep_combine(
                expert_out,
                async_finish=True,
            )
            outputs.append(dispatcher.finish_deepep_combine(combine_state))
        return torch.cat(outputs, dim=0)


__all__ = ["EPChunkedMoELayer"]
