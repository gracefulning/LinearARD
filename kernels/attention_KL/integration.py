import torch
from .kernel_A import lse_stats_triton
from .kernel_B import kl_forward_triton
from .kernel_C import kl_backward_triton

class AttnKLAlignFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qs, ks, qt, kt,
                attn_mask, causal: bool,
                sm_scale_s, sm_scale_t):
        """
        qs, ks, qt, kt: [B, H, T, D]
        attn_mask: [B, T], 1/True means a valid token; None means no padding
        Returns: scalar loss
        """

        # 1) Compute per-row logsumexp
        lse_s = lse_stats_triton(qs, ks, attn_mask=attn_mask, causal=causal, sm_scale=sm_scale_s)
        lse_t = lse_stats_triton(qt, kt, attn_mask=attn_mask, causal=causal, sm_scale=sm_scale_t)

        # 2) forward KL
        kl = kl_forward_triton(
            qs, ks, lse_s,
            qt, kt, lse_t,
            attn_mask=attn_mask, causal=causal,
            sm_scale_s=sm_scale_s, sm_scale_t=sm_scale_t,
        )
        if kl.ndim != 0:
            kl = kl.sum()

        # 3) Save tensors required by backward
        # Teacher has no gradient output, but backward needs qt/kt/lse_t to recompute pt
        if attn_mask is None:
            dummy_mask = torch.empty(0, device=qs.device)
            has_mask = False
        else:
            dummy_mask = attn_mask
            has_mask = True

        ctx.save_for_backward(qs, ks, lse_s, qt, kt, lse_t, dummy_mask)
        ctx.has_mask = has_mask
        ctx.causal = causal
        ctx.sm_scale_s = sm_scale_s
        ctx.sm_scale_t = sm_scale_t
        return kl

    @staticmethod
    def backward(ctx, dloss):
        qs, ks, lse_s, qt, kt, lse_t, mask_or_empty = ctx.saved_tensors
        attn_mask = mask_or_empty if ctx.has_mask else None

        # Backward: call the Triton backward kernel
        dqs, dks = kl_backward_triton(
            qs, ks, lse_s,
            qt, kt, lse_t,
            attn_mask=attn_mask,
            causal=ctx.causal,
            sm_scale_s=ctx.sm_scale_s,
            sm_scale_t=ctx.sm_scale_t,
        )

        # Chain rule
        dqs = dqs * dloss
        dks = dks * dloss

        # No gradient for teacher tensors; no gradient for mask/causal/scale either
        return dqs, dks, None, None, None, None, None, None


def attn_kl_align(qs, ks, qt, kt,
                 attn_mask=None, causal=True,
                 sm_scale_s=None, sm_scale_t=None):
    return AttnKLAlignFn.apply(qs, ks, qt, kt, attn_mask, causal, sm_scale_s, sm_scale_t)


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skip attn_kl_align smoke test")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    B, H, T, D = 1, 2, 8, 16
    sm_scale = 1.0 / (D ** 0.5)
    attn_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], device=device, dtype=torch.int32)

    qs = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    ks = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
    qt = torch.randn(B, H, T, D, device=device, dtype=dtype)
    kt = torch.randn(B, H, T, D, device=device, dtype=dtype)

    loss_custom = attn_kl_align(
        qs, ks, qt, kt,
        attn_mask=attn_mask,
        causal=True,
        sm_scale_s=sm_scale,
        sm_scale_t=sm_scale,
    )
    loss_custom.backward()

    qs_f = qs.detach().float()
    ks_f = ks.detach().float()
    qt_f = qt.detach().float()
    kt_f = kt.detach().float()
    logits_s = torch.matmul(qs_f, ks_f.transpose(-1, -2)) * sm_scale
    logits_t = torch.matmul(qt_f, kt_f.transpose(-1, -2)) * sm_scale

    row_mask = attn_mask[:, None, :, None].bool()
    col_mask = attn_mask[:, None, None, :].bool()
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))[None, None, :, :]
    valid = row_mask & col_mask & causal_mask

    neg_inf = torch.finfo(logits_s.dtype).min
    logits_s = logits_s.masked_fill(~valid, neg_inf)
    logits_t = logits_t.masked_fill(~valid, neg_inf)

    valid_rows = valid.any(dim=-1)
    mask_rows = valid_rows.unsqueeze(-1)
    log_probs_s = torch.log_softmax(logits_s, dim=-1)
    log_probs_t = torch.log_softmax(logits_t, dim=-1)
    probs_t = torch.softmax(logits_t, dim=-1)
    log_probs_s = torch.where(mask_rows, log_probs_s, torch.zeros_like(log_probs_s))
    log_probs_t = torch.where(mask_rows, log_probs_t, torch.zeros_like(log_probs_t))
    probs_t = torch.where(mask_rows, probs_t, torch.zeros_like(probs_t))

    loss_ref = (probs_t * (log_probs_t - log_probs_s)).sum()

    print("custom loss:", loss_custom.item())
    print("ref loss:", loss_ref.item())
    print("abs diff:", float((loss_custom.float() - loss_ref).abs().item()))
    print("qs grad max:", qs.grad.abs().max().item(), "ks grad max:", ks.grad.abs().max().item())


if __name__ == "__main__":
    main()
