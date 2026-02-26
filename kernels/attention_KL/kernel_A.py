import math

import torch
import triton
import triton.language as tl

# Kernel A: LSE computation
# Interface:
# - lse_stats_triton(q, k, attn_mask=None, causal=True, sm_scale=None)
# - q: [B,Hq,T,D] CUDA fp16/bf16; k: [B,Hkv,T,D] CUDA fp16/bf16
# - Requires Hq % Hkv == 0, with mapping kv_head = h // (Hq/Hkv)
# - attn_mask: [B,T] (CPU/CUDA) bool/int32/int64/uint8, where 1 means valid token
# - Returns: lse [B,Hq,T] CUDA float32
# Tunable parameters:
# - SUPPORTED_HEAD_DIMS: supported head_dim list
# - _BLOCK_M/_BLOCK_N, _NUM_WARPS/_NUM_STAGES: Triton tile/parallel settings

SUPPORTED_HEAD_DIMS = (8, 16, 32, 64, 128, 256)


@triton.jit(
    do_not_specialize=[
        "stride_qb",
        "stride_qh",
        "stride_qt",
        "stride_qd",
        "stride_kb",
        "stride_kh",
        "stride_kt",
        "stride_kd",
        "stride_mb",
        "stride_mt",
        "stride_ob",
        "stride_oh",
        "stride_ot",
        "B",
        "Hq",
        "G",
        "sm_scale",
    ]
)
def _lse_stats_kernel(
    q_ptr,
    k_ptr,
    mask_ptr,
    lse_ptr,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_mb,
    stride_mt,
    stride_ob,
    stride_oh,
    stride_ot,
    B,
    Hq,
    T,
    G,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # Program IDs: dim 0 maps to (B*Hq), dim 1 maps to row tiles
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // Hq
    h = pid_bh - b * Hq
    kv_head = h // G

    # Row and column indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    # Base pointers
    q_base = q_ptr + b * stride_qb + h * stride_qh
    k_base = k_ptr + b * stride_kb + kv_head * stride_kh
    out_base = lse_ptr + b * stride_ob + h * stride_oh
    mask_base = mask_ptr + b * stride_mb

    # Row validity
    mask_i = offs_m < T
    if HAS_MASK:
        mask_q = tl.load(mask_base + offs_m * stride_mt, mask=mask_i, other=0)
        mask_q = mask_q != 0
        valid_rows = mask_i & mask_q
    else:
        valid_rows = mask_i

    # Load Q (keep original dtype; cast to float32 only when needed)
    q = tl.load(
        q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd,
        mask=mask_i[:, None],
        other=0.0,
    )

    # Initialize online log-sum-exp
    m = tl.full([BLOCK_M], -float("inf"), tl.float32)
    r = tl.zeros([BLOCK_M], tl.float32)

    # Iterate over key column tiles
    for start_n in range(0, T, BLOCK_N):
        j = start_n + offs_n
        mask_j = j < T

        if HAS_MASK:
            mask_k = tl.load(mask_base + j * stride_mt, mask=mask_j, other=0)
            mask_k = mask_k != 0
            valid_cols = mask_j & mask_k
        else:
            valid_cols = mask_j

        # Load K (keep original dtype; cast to float32 only when needed)
        k = tl.load(
            k_base + j[:, None] * stride_kt + offs_d[None, :] * stride_kd,
            mask=mask_j[:, None],
            other=0.0,
        )

        # Compute logits via dot product (for D=8, use elementwise reduction to avoid tl.dot limits)
        if D < 16:
            q_f = q.to(tl.float32)
            k_f = k.to(tl.float32)
            scores = tl.sum(q_f[:, None, :] * k_f[None, :, :], axis=2) * sm_scale
        else:
            scores = tl.dot(q, tl.trans(k)) * sm_scale
            scores = scores.to(tl.float32)

        # Compose validity mask (padding + causal)
        valid = valid_rows[:, None] & valid_cols[None, :]
        if CAUSAL:
            valid = valid & (j[None, :] <= offs_m[:, None])

        scores = tl.where(valid, scores, -float("inf"))

        # Block-wise max and sum
        m_blk = tl.max(scores, axis=1)
        m_blk_safe = tl.where(m_blk == -float("inf"), 0.0, m_blk)
        exp_scores = tl.where(valid, tl.exp(scores - m_blk_safe[:, None]), 0.0)
        r_blk = tl.sum(exp_scores, axis=1)

        # Online merge
        m_new = tl.maximum(m, m_blk)
        r = r * tl.exp(m - m_new) + r_blk * tl.exp(m_blk - m_new)
        m = m_new

    # Output LSE: write 0 for invalid query rows
    lse = tl.log(r) + m
    out = tl.where(valid_rows, lse, 0.0)
    tl.store(out_base + offs_m * stride_ot, out, mask=mask_i)


# Fixed compile-time parameters (no autotune)
_BLOCK_M = 32
_BLOCK_N = 64
_NUM_WARPS = 4
_NUM_STAGES = 2


def get_lse_kernel_cache_size():
    """Return the JIT cache entry count for this kernel."""
    if hasattr(_lse_stats_kernel, "cache"):
        cache = _lse_stats_kernel.cache
    elif hasattr(_lse_stats_kernel, "_cache"):
        cache = _lse_stats_kernel._cache
    else:
        raise RuntimeError("Cannot locate Triton JIT cache container")
    keys = []
    if isinstance(cache, dict):
        stack = [cache]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                if cur and all(not isinstance(v, dict) for v in cur.values()):
                    keys.extend(list(cur.keys()))
                else:
                    stack.extend(cur.values())
    return len(keys)


def lse_stats_triton(q, k, attn_mask=None, causal=True, sm_scale=None):
    """Triton kernel wrapper."""
    assert isinstance(q, torch.Tensor), "q must be a torch.Tensor"
    assert isinstance(k, torch.Tensor), "k must be a torch.Tensor"
    assert q.is_cuda and k.is_cuda, "q/k must be on CUDA"
    assert q.dtype in (torch.float16, torch.bfloat16), "q dtype supports only fp16/bf16"
    assert k.dtype in (torch.float16, torch.bfloat16), "k dtype supports only fp16/bf16"
    assert q.dtype == k.dtype, "q/k dtype must match"
    assert q.ndim == 4 and k.ndim == 4, "q/k shape must be [B,H,T,D]"
    assert q.shape[0] == k.shape[0], "q/k B must match"
    assert q.shape[2] == k.shape[2], "q/k T must match"
    assert q.shape[3] == k.shape[3], "q/k D must match"
    assert q.is_contiguous() and k.is_contiguous(), "q/k must be contiguous"
    assert q.stride(-1) == 1 and k.stride(-1) == 1, "q/k last-dim stride must be 1"

    B, Hq, T, D = q.shape
    Hkv = k.shape[1]
    assert D in SUPPORTED_HEAD_DIMS, "D is not in the supported set"
    assert Hq % Hkv == 0, "Hq must be divisible by Hkv"

    has_mask = attn_mask is not None
    if has_mask:
        assert isinstance(attn_mask, torch.Tensor), "attn_mask must be a torch.Tensor"
        if attn_mask.device.type == "cpu":
            attn_mask = attn_mask.to(device=q.device)
        assert attn_mask.is_cuda, "attn_mask must be on CUDA"
        assert attn_mask.ndim == 2, "attn_mask shape must be [B,T]"
        assert attn_mask.shape[0] == B and attn_mask.shape[1] == T, "attn_mask shape must be [B,T]"
        assert attn_mask.dtype in (torch.bool, torch.int32, torch.int64, torch.uint8), "attn_mask dtype is invalid"
        assert attn_mask.is_contiguous(), "attn_mask must be contiguous"
        assert attn_mask.stride(-1) == 1, "attn_mask last-dim stride must be 1"
    else:
        attn_mask = q  # Placeholder pointer; not used in the kernel

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    assert isinstance(sm_scale, float) or isinstance(sm_scale, int), "sm_scale must be a scalar"
    sm_scale = float(sm_scale)

    g = Hq // Hkv
    lse = torch.empty((B, Hq, T), device=q.device, dtype=torch.float32)

    grid = (B * Hq, triton.cdiv(T, _BLOCK_M))
    _lse_stats_kernel[grid](
        q,
        k,
        attn_mask,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        attn_mask.stride(0),
        attn_mask.stride(1),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        B,
        Hq,
        T,
        g,
        sm_scale,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        D=D,
        CAUSAL=causal,
        HAS_MASK=has_mask,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return lse
