import math

import torch
import triton
import triton.language as tl

# Kernel C: KL(Pt||Ps) backward gradients
# Interface:
# - kl_backward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=True, sm_scale_s=None, sm_scale_t=None)
# - qs: [B,Hq,T,Ds]; ks: [B,Hkv_s,T,Ds]
# - qt: [B,Hq,T,Dt]; kt: [B,Hkv_t,T,Dt]
# - lse_s/lse_t: [B,Hq,T] CUDA float32
# - Returns: dq [B,Hq,T,Ds]; dk [B,Hkv_s,T,Ds]
# Tunable parameters:
# - SUPPORTED_HEAD_DIMS: supported head_dim list
# - _BLOCK_M/_BLOCK_N/_BLOCK_D, _NUM_WARPS/_NUM_STAGES: Triton tile/parallel settings

SUPPORTED_HEAD_DIMS = (8, 16, 32, 64, 128, 256)


@triton.jit(
    do_not_specialize=[
        "stride_qsb",
        "stride_qsh",
        "stride_qst",
        "stride_qsd",
        "stride_ksb",
        "stride_ksh",
        "stride_kst",
        "stride_ksd",
        "stride_qtb",
        "stride_qth",
        "stride_qtt",
        "stride_qtd",
        "stride_ktb",
        "stride_kth",
        "stride_ktt",
        "stride_ktd",
        "stride_lseb",
        "stride_lseh",
        "stride_lset",
        "stride_lteb",
        "stride_lteh",
        "stride_ltet",
        "stride_mb",
        "stride_mt",
        "stride_dqb",
        "stride_dqh",
        "stride_dqt",
        "stride_dqd",
        "stride_dkb",
        "stride_dkh",
        "stride_dkt",
        "stride_dkd",
        "B",
        "Hq",
        "Gs",
        "Gt",
        "sm_scale_s",
        "sm_scale_t",
    ]
)
def _kl_backward_kernel(
    qs_ptr,
    ks_ptr,
    qt_ptr,
    kt_ptr,
    lse_s_ptr,
    lse_t_ptr,
    mask_ptr,
    dq_ptr,
    dk_ptr,
    stride_qsb,
    stride_qsh,
    stride_qst,
    stride_qsd,
    stride_ksb,
    stride_ksh,
    stride_kst,
    stride_ksd,
    stride_qtb,
    stride_qth,
    stride_qtt,
    stride_qtd,
    stride_ktb,
    stride_kth,
    stride_ktt,
    stride_ktd,
    stride_lseb,
    stride_lseh,
    stride_lset,
    stride_lteb,
    stride_lteh,
    stride_ltet,
    stride_mb,
    stride_mt,
    stride_dqb,
    stride_dqh,
    stride_dqt,
    stride_dqd,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    B,
    Hq,
    T,
    Gs,
    Gt,
    sm_scale_s,
    sm_scale_t,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    Ds: tl.constexpr,
    Dt: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    # Program IDs: dim 0 maps to (B*Hq), dim 1 maps to row tiles
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // Hq
    h = pid_bh - b * Hq
    kv_head_s = h // Gs
    kv_head_t = h // Gt

    # Row, column, and D indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_ds = tl.arange(0, Ds)
    offs_dt = tl.arange(0, Dt)

    # Base pointers
    qs_base = qs_ptr + b * stride_qsb + h * stride_qsh
    ks_base = ks_ptr + b * stride_ksb + kv_head_s * stride_ksh
    qt_base = qt_ptr + b * stride_qtb + h * stride_qth
    kt_base = kt_ptr + b * stride_ktb + kv_head_t * stride_kth
    lse_s_base = lse_s_ptr + b * stride_lseb + h * stride_lseh
    lse_t_base = lse_t_ptr + b * stride_lteb + h * stride_lteh
    mask_base = mask_ptr + b * stride_mb
    dq_base = dq_ptr + b * stride_dqb + h * stride_dqh
    dk_base = dk_ptr + b * stride_dkb + kv_head_s * stride_dkh

    # Row validity
    mask_i = offs_m < T
    if HAS_MASK:
        mask_q = tl.load(mask_base + offs_m * stride_mt, mask=mask_i, other=0)
        mask_q = mask_q != 0
        valid_rows = mask_i & mask_q
    else:
        valid_rows = mask_i

    # Load LSE and mask out empty-set rows
    lse_s_row = tl.load(lse_s_base + offs_m * stride_lset, mask=mask_i, other=0.0)
    lse_t_row = tl.load(lse_t_base + offs_m * stride_ltet, mask=mask_i, other=0.0)
    row_has_lse = (lse_s_row != -float("inf")) & (lse_t_row != -float("inf"))
    valid_rows = valid_rows & row_has_lse
    lse_s_safe = tl.where(lse_s_row == -float("inf"), 0.0, lse_s_row)
    lse_t_safe = tl.where(lse_t_row == -float("inf"), 0.0, lse_t_row)

    # Load Q (student/teacher)
    qs = tl.load(
        qs_base + offs_m[:, None] * stride_qst + offs_ds[None, :] * stride_qsd,
        mask=mask_i[:, None],
        other=0.0,
    )
    qt = tl.load(
        qt_base + offs_m[:, None] * stride_qtt + offs_dt[None, :] * stride_qtd,
        mask=mask_i[:, None],
        other=0.0,
    )

    # Row-wise dq accumulator
    dq_acc = tl.zeros([BLOCK_M, Ds], tl.float32)

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

        # Load K (student/teacher)
        ks = tl.load(
            ks_base + j[:, None] * stride_kst + offs_ds[None, :] * stride_ksd,
            mask=mask_j[:, None],
            other=0.0,
        )
        kt = tl.load(
            kt_base + j[:, None] * stride_ktt + offs_dt[None, :] * stride_ktd,
            mask=mask_j[:, None],
            other=0.0,
        )

        # Compute logits via dot product (for Ds/Dt=8, use elementwise reduction to avoid tl.dot limits)
        if Ds < 16:
            qs_f = qs.to(tl.float32)
            ks_f = ks.to(tl.float32)
            scores_s = tl.sum(qs_f[:, None, :] * ks_f[None, :, :], axis=2) * sm_scale_s
        else:
            scores_s = tl.dot(qs, tl.trans(ks)) * sm_scale_s
            scores_s = scores_s.to(tl.float32)
        if Dt < 16:
            qt_f = qt.to(tl.float32)
            kt_f = kt.to(tl.float32)
            scores_t = tl.sum(qt_f[:, None, :] * kt_f[None, :, :], axis=2) * sm_scale_t
        else:
            scores_t = tl.dot(qt, tl.trans(kt)) * sm_scale_t
            scores_t = scores_t.to(tl.float32)

        # Validity mask (padding + causal + bounds)
        valid = valid_rows[:, None] & valid_cols[None, :]
        if CAUSAL:
            valid = valid & (j[None, :] <= offs_m[:, None])

        # Compute logps/logpt and ps/pt (set invalid positions to 0)
        logps = scores_s - lse_s_safe[:, None]
        logpt = scores_t - lse_t_safe[:, None]
        logps = tl.where(valid, logps, 0.0)
        logpt = tl.where(valid, logpt, 0.0)
        ps = tl.where(valid, tl.exp(logps), 0.0)
        pt = tl.where(valid, tl.exp(logpt), 0.0)

        # D = ps - pt
        dval = ps - pt
        dval_cast = dval.to(qs.dtype)

        # dq accumulation: alpha_s * (D @ Ks); for Ds=8 use elementwise reduction to avoid tl.dot limits
        if Ds < 16:
            dq_acc += sm_scale_s * tl.sum(dval_cast[:, :, None] * ks[None, :, :], axis=1)
        else:
            dq_acc += sm_scale_s * tl.dot(dval_cast, ks)

        # dk accumulation: alpha_s * (D^T @ Qs), written back via atomic_add
        for d_start in tl.static_range(0, Ds, BLOCK_D):
            offs_db = d_start + tl.arange(0, BLOCK_D)
            mask_db = offs_db < Ds
            qs_block = tl.load(
                qs_base + offs_m[:, None] * stride_qst + offs_db[None, :] * stride_qsd,
                mask=mask_i[:, None] & mask_db[None, :],
                other=0.0,
            )
            dk_block = sm_scale_s * tl.dot(tl.trans(dval_cast), qs_block)
            dk_ptrs = dk_base + j[:, None] * stride_dkt + offs_db[None, :] * stride_dkd
            mask_store = valid_cols[:, None] & mask_db[None, :]
            tl.atomic_add(dk_ptrs, dk_block, mask=mask_store)

    # Zero invalid rows
    dq_acc = tl.where(valid_rows[:, None], dq_acc, 0.0)

    # Store dq
    dq_ptrs = dq_base + offs_m[:, None] * stride_dqt + offs_ds[None, :] * stride_dqd
    mask_out = mask_i[:, None]
    tl.store(dq_ptrs, dq_acc, mask=mask_out)


# Fixed compile-time parameters (autotune disabled)
_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_D = 32
_NUM_WARPS = 4
_NUM_STAGES = 2


def get_kl_backward_cache_size():
    """Return the JIT cache entry count for this kernel."""
    if hasattr(_kl_backward_kernel, "cache"):
        cache = _kl_backward_kernel.cache
    elif hasattr(_kl_backward_kernel, "_cache"):
        cache = _kl_backward_kernel._cache
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


def kl_backward_triton(
    qs,
    ks,
    lse_s,
    qt,
    kt,
    lse_t,
    attn_mask=None,
    causal=True,
    sm_scale_s=None,
    sm_scale_t=None,
):
    """Triton kernel wrapper for KL backward gradients."""
    # Basic type and device checks
    assert isinstance(qs, torch.Tensor), "qs must be a torch.Tensor"
    assert isinstance(ks, torch.Tensor), "ks must be a torch.Tensor"
    assert isinstance(qt, torch.Tensor), "qt must be a torch.Tensor"
    assert isinstance(kt, torch.Tensor), "kt must be a torch.Tensor"
    assert isinstance(lse_s, torch.Tensor), "lse_s must be a torch.Tensor"
    assert isinstance(lse_t, torch.Tensor), "lse_t must be a torch.Tensor"

    assert qs.is_cuda and ks.is_cuda and qt.is_cuda and kt.is_cuda, "Q/K must be on CUDA"
    assert lse_s.is_cuda and lse_t.is_cuda, "lse must be on CUDA"

    assert qs.dtype in (torch.float16, torch.bfloat16), "qs dtype supports only fp16/bf16"
    assert ks.dtype in (torch.float16, torch.bfloat16), "ks dtype supports only fp16/bf16"
    assert qt.dtype in (torch.float16, torch.bfloat16), "qt dtype supports only fp16/bf16"
    assert kt.dtype in (torch.float16, torch.bfloat16), "kt dtype supports only fp16/bf16"
    assert qs.dtype == ks.dtype, "qs/ks dtype must match"
    assert qt.dtype == kt.dtype, "qt/kt dtype must match"

    assert qs.ndim == 4 and ks.ndim == 4, "qs/ks shape must be [B,H,T,Ds]"
    assert qt.ndim == 4 and kt.ndim == 4, "qt/kt shape must be [B,H,T,Dt]"
    assert qs.shape[0] == ks.shape[0], "qs/ks B must match"
    assert qs.shape[2] == ks.shape[2], "qs/ks T must match"
    assert qs.shape[3] == ks.shape[3], "qs/ks Ds must match"
    assert qt.shape[0] == kt.shape[0], "qt/kt B must match"
    assert qt.shape[2] == kt.shape[2], "qt/kt T must match"
    assert qt.shape[3] == kt.shape[3], "qt/kt Dt must match"
    assert lse_s.ndim == 3 and lse_t.ndim == 3, "lse shape must be [B,H,T]"

    assert qs.is_contiguous() and ks.is_contiguous(), "qs/ks must be contiguous"
    assert qt.is_contiguous() and kt.is_contiguous(), "qt/kt must be contiguous"
    assert qs.stride(-1) == 1 and ks.stride(-1) == 1, "qs/ks last-dim stride must be 1"
    assert qt.stride(-1) == 1 and kt.stride(-1) == 1, "qt/kt last-dim stride must be 1"

    assert lse_s.dtype == torch.float32 and lse_t.dtype == torch.float32, "lse must be float32"
    assert lse_s.is_contiguous() and lse_t.is_contiguous(), "lse must be contiguous"
    assert lse_s.stride(-1) == 1 and lse_t.stride(-1) == 1, "lse last-dim stride must be 1"

    B, Hq, T, Ds = qs.shape
    B2, Hq2, T2, Dt = qt.shape
    Hkv_s = ks.shape[1]
    Hkv_t = kt.shape[1]
    assert (B, Hq, T) == lse_s.shape, "lse_s shape must match [B,Hq,T]"
    assert (B2, Hq2, T2) == lse_t.shape, "lse_t shape must match [B,Hq,T]"
    assert (B, Hq, T) == (B2, Hq2, T2), "student/teacher B/Hq/T must align"
    assert Hq % Hkv_s == 0, "Hq must be divisible by Hkv_s"
    assert Hq % Hkv_t == 0, "Hq must be divisible by Hkv_t"

    assert Ds in SUPPORTED_HEAD_DIMS, "Ds is not in the supported set"
    assert Dt in SUPPORTED_HEAD_DIMS, "Dt is not in the supported set"

    # Mask checks (CPU or CUDA input is allowed)
    has_mask = attn_mask is not None
    if has_mask:
        assert isinstance(attn_mask, torch.Tensor), "attn_mask must be a torch.Tensor"
        if attn_mask.device.type == "cpu":
            assert attn_mask.is_contiguous(), "attn_mask must be contiguous"
            attn_mask = attn_mask.to(device=qs.device)
        assert attn_mask.is_cuda, "attn_mask must be on CUDA"
        assert attn_mask.ndim == 2, "attn_mask shape must be [B,T]"
        assert attn_mask.shape[0] == B and attn_mask.shape[1] == T, "attn_mask shape must be [B,T]"
        assert attn_mask.dtype in (torch.bool, torch.int32, torch.int64, torch.uint8), "attn_mask dtype is invalid"
        assert attn_mask.is_contiguous(), "attn_mask must be contiguous"
        assert attn_mask.stride(-1) == 1, "attn_mask last-dim stride must be 1"
    else:
        attn_mask = qs  # Placeholder pointer; not used in the kernel

    # Scale handling
    if sm_scale_s is None:
        sm_scale_s = 1.0 / math.sqrt(Ds)
    if sm_scale_t is None:
        sm_scale_t = 1.0 / math.sqrt(Dt)
    assert isinstance(sm_scale_s, float) or isinstance(sm_scale_s, int), "sm_scale_s must be a scalar"
    assert isinstance(sm_scale_t, float) or isinstance(sm_scale_t, int), "sm_scale_t must be a scalar"
    sm_scale_s = float(sm_scale_s)
    sm_scale_t = float(sm_scale_t)

    # Output gradients (float32), initialized to zeros
    dq = torch.zeros((B, Hq, T, Ds), device=qs.device, dtype=torch.float32)
    dk = torch.zeros((B, Hkv_s, T, Ds), device=qs.device, dtype=torch.float32)

    g_s = Hq // Hkv_s
    g_t = Hq // Hkv_t
    grid = (B * Hq, triton.cdiv(T, _BLOCK_M))
    _kl_backward_kernel[grid](
        qs,
        ks,
        qt,
        kt,
        lse_s,
        lse_t,
        attn_mask,
        dq,
        dk,
        qs.stride(0),
        qs.stride(1),
        qs.stride(2),
        qs.stride(3),
        ks.stride(0),
        ks.stride(1),
        ks.stride(2),
        ks.stride(3),
        qt.stride(0),
        qt.stride(1),
        qt.stride(2),
        qt.stride(3),
        kt.stride(0),
        kt.stride(1),
        kt.stride(2),
        kt.stride(3),
        lse_s.stride(0),
        lse_s.stride(1),
        lse_s.stride(2),
        lse_t.stride(0),
        lse_t.stride(1),
        lse_t.stride(2),
        attn_mask.stride(0),
        attn_mask.stride(1),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        B,
        Hq,
        T,
        g_s,
        g_t,
        sm_scale_s,
        sm_scale_t,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=_BLOCK_D,
        Ds=Ds,
        Dt=Dt,
        CAUSAL=causal,
        HAS_MASK=has_mask,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    return dq, dk
