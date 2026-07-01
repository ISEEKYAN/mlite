import pytest
import torch

from megatron.lite.primitive.kernels.rope import (
    apply_fused_mla_rotary_for_kv,
    apply_fused_mla_rotary_for_q,
    apply_fused_rotary,
)
from megatron.lite.primitive.utils.rope import (
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class _CP1:
    @staticmethod
    def size():
        return 1

    @staticmethod
    def rank():
        return 0


def _freqs(seq: int, dim: int) -> torch.Tensor:
    inv = 1.0 / (10_000 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    phase = torch.outer(torch.arange(seq, device="cuda").float(), inv)
    return torch.cat([phase, phase], dim=-1)[:, None, None, :]


@pytest.mark.parametrize("packed", [False, True])
def test_generic_fused_rope_forward_backward(packed):
    torch.manual_seed(7)
    freqs = _freqs(4 if packed else 8, 16)
    if packed:
        cu = torch.tensor([0, 4, 8], device="cuda", dtype=torch.int32)
        source = torch.randn(8, 3, 16, device="cuda", dtype=torch.bfloat16)
        ref_input = source.detach().clone().requires_grad_(True)
        fused_input = source.detach().clone().requires_grad_(True)
        ref = _apply_rotary_pos_emb_thd(ref_input, cu, freqs, cp_group=_CP1())
        fused = apply_fused_rotary(fused_input, freqs, cu_seqlens=cu)
    else:
        source = torch.randn(8, 1, 3, 16, device="cuda", dtype=torch.bfloat16)
        ref_input = source.detach().clone().requires_grad_(True)
        fused_input = source.detach().clone().requires_grad_(True)
        ref = _apply_rotary_pos_emb_bshd(ref_input, freqs)
        fused = apply_fused_rotary(fused_input, freqs)
    torch.testing.assert_close(fused, ref, rtol=0, atol=2e-2)
    grad = torch.randn_like(ref)
    ref.backward(grad)
    fused.backward(grad)
    torch.testing.assert_close(fused_input.grad, ref_input.grad, rtol=0, atol=2e-2)


def test_mla_q_and_kv_fused_rope_forward_backward():
    torch.manual_seed(11)
    seq, heads, nope, rope, value = 8, 4, 8, 8, 16
    phase = _freqs(seq, rope)
    cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
    q = torch.randn(seq, 1, heads, nope + rope, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(seq, 1, heads, nope + value, device="cuda", dtype=torch.bfloat16)
    k_pos = torch.randn(seq, 1, 1, rope, device="cuda", dtype=torch.bfloat16)
    q_ref = q.detach().clone().requires_grad_(True)
    q_fused = q.detach().clone().requires_grad_(True)
    kv_ref = kv.detach().clone().requires_grad_(True)
    kv_fused = kv.detach().clone().requires_grad_(True)
    kp_ref = k_pos.detach().clone().requires_grad_(True)
    kp_fused = k_pos.detach().clone().requires_grad_(True)

    q_nope, q_pos = q_ref.split([nope, rope], dim=-1)
    q_pos = _apply_rotary_pos_emb_bshd(q_pos, phase, mla_rotary_interleaved=True)
    expected_q = torch.cat([q_nope, q_pos], dim=-1)
    expected_k = torch.cat(
        [
            kv_ref[..., :nope],
            _apply_rotary_pos_emb_bshd(
                kp_ref, phase, mla_rotary_interleaved=True
            ).expand(seq, 1, heads, rope),
        ],
        dim=-1,
    )
    expected_v = kv_ref[..., nope:]

    actual_q = apply_fused_mla_rotary_for_q(
        q_fused.clone(), cos, sin, nope_dim=nope, rope_dim=rope
    )
    actual_k, actual_v = apply_fused_mla_rotary_for_kv(
        kv_fused,
        kp_fused,
        cos,
        sin,
        rope_dim=rope,
        key_nope_dim=nope,
        value_dim=value,
    )
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=2e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=2e-2)
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=2e-2)
    (expected_q.sum() + expected_k.sum() + expected_v.sum()).backward()
    (actual_q.sum() + actual_k.sum() + actual_v.sum()).backward()
    torch.testing.assert_close(q_fused.grad, q_ref.grad, rtol=0, atol=2e-2)
    torch.testing.assert_close(kv_fused.grad, kv_ref.grad, rtol=0, atol=2e-2)
    # The fused kernel reduces this shared positional gradient across heads in
    # BF16; allow the corresponding reduction-order rounding while retaining
    # the same absolute bound used by the other MLA checks.
    torch.testing.assert_close(kp_fused.grad, kp_ref.grad, rtol=1e-2, atol=2e-2)


def test_mla_inverse_remove_interleaving_round_trip():
    seq, heads, dim = 8, 2, 8
    phase = _freqs(seq, dim)
    cos, sin = phase.cos().contiguous(), phase.sin().contiguous()
    source = torch.randn(seq, 1, heads, dim, device="cuda", dtype=torch.bfloat16)
    rotated = apply_fused_mla_rotary_for_q(
        source.clone(), cos, sin, nope_dim=0, rope_dim=dim, remove_interleaving=True
    )
    restored = apply_fused_mla_rotary_for_q(
        rotated.clone(),
        cos,
        sin,
        nope_dim=0,
        rope_dim=dim,
        inverse=True,
        remove_interleaving=True,
    )
    torch.testing.assert_close(restored, source, rtol=0, atol=3e-2)


def test_dsa_and_csa_site_adapters_forward_backward():
    from megatron.lite.primitive.modules.attention.csa import (
        _apply_partial_rope_dispatch,
        apply_partial_rope,
    )
    from megatron.lite.primitive.modules.attention.dsa import (
        _apply_fused_dsa_rope,
        apply_rotary_pos_emb,
    )

    phase = _freqs(8, 8).squeeze(1).squeeze(1).unsqueeze(0)
    cos, sin = phase.cos().to(torch.bfloat16), phase.sin().to(torch.bfloat16)
    source = torch.randn(1, 8, 3, 8, device="cuda", dtype=torch.bfloat16)
    dsa_ref_input = source.detach().clone().requires_grad_(True)
    dsa_fused_input = source.detach().clone().requires_grad_(True)
    dsa_ref = apply_rotary_pos_emb(dsa_ref_input, cos, sin, unsqueeze_dim=2)
    dsa_fused = _apply_fused_dsa_rope(dsa_fused_input, phase)
    torch.testing.assert_close(dsa_fused, dsa_ref, rtol=0, atol=2e-2)
    dsa_ref.sum().backward()
    dsa_fused.sum().backward()
    torch.testing.assert_close(dsa_fused_input.grad, dsa_ref_input.grad, rtol=0, atol=2e-2)

    csa_source = source.transpose(1, 2).detach()
    csa_ref_input = csa_source.clone().requires_grad_(True)
    csa_fused_input = csa_source.clone().requires_grad_(True)
    csa_ref = apply_partial_rope(csa_ref_input, cos, sin, 8)
    csa_fused = _apply_partial_rope_dispatch(
        csa_fused_input, cos, sin, 8, fused=True
    )
    torch.testing.assert_close(csa_fused, csa_ref, rtol=0, atol=2e-2)
    csa_ref.sum().backward()
    csa_fused.sum().backward()
    torch.testing.assert_close(csa_fused_input.grad, csa_ref_input.grad, rtol=0, atol=2e-2)
