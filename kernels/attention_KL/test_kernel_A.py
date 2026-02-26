import inspect
import os
import subprocess
import sys
import threading
import time

import torch

# Ensure imports work from the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from kernels.attention_KL import kernel_A


def _collect_leaf_keys(obj, out):
    if isinstance(obj, dict):
        if obj and all(not isinstance(v, dict) for v in obj.values()):
            out.extend(list(obj.keys()))
        else:
            for v in obj.values():
                _collect_leaf_keys(v, out)


def _get_cache_leaf_keys(jit_fn):
    if hasattr(jit_fn, "cache"):
        cache = jit_fn.cache
    elif hasattr(jit_fn, "_cache"):
        cache = jit_fn._cache
    elif hasattr(jit_fn, "device_caches"):
        keys = []
        for entry in jit_fn.device_caches.values():
            if isinstance(entry, dict):
                _collect_leaf_keys(entry, keys)
            elif isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], dict):
                _collect_leaf_keys(entry[0], keys)
        return keys
    else:
        raise RuntimeError("Cannot locate Triton JIT cache container")
    keys = []
    _collect_leaf_keys(cache, keys)
    return keys


def _cache_leaf_count(jit_fn):
    return len(_get_cache_leaf_keys(jit_fn))


def _print_cache_growth(label, before_keys, after_keys):
    new_keys = [k for k in after_keys if k not in before_keys]
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(
        f"timestamp={now} | {label} | before={len(before_keys)} after={len(after_keys)} new={len(new_keys)}"
    )
    if new_keys:
        print(f"{label} new_keys_sample={new_keys[:5]}")


# Reference implementation
def reference_lse(q, k, attn_mask, causal, sm_scale):
    q_f = q.float()
    k_f = k.float()
    B, Hq, T, _ = q_f.shape
    Hkv = k_f.shape[1]
    g = Hq // Hkv

    kv_map = torch.arange(Hq, device=q.device) // g
    k_map = k_f[:, kv_map, :, :]
    scores = torch.matmul(q_f, k_map.transpose(-1, -2)) * sm_scale

    if attn_mask is not None:
        mask_bool = attn_mask.to(torch.bool)
        valid_k = mask_bool[:, None, None, :]
    else:
        valid_k = torch.ones((B, 1, 1, T), device=q.device, dtype=torch.bool)

    if causal:
        causal_mask = torch.tril(torch.ones((T, T), device=q.device, dtype=torch.bool))
        valid = valid_k & causal_mask[None, None, :, :]
    else:
        valid = valid_k

    scores = torch.where(valid, scores, torch.tensor(float("-inf"), device=q.device))

    row_max = scores.max(dim=-1).values
    row_max_safe = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max_safe[..., None])
    exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
    row_sum = exp_scores.sum(dim=-1)
    lse = row_max + torch.log(row_sum)

    if attn_mask is not None:
        valid_q = attn_mask.to(torch.bool)[:, None, :]
        lse = torch.where(valid_q, lse, torch.zeros_like(lse))

    return lse



def static_audit_wrapper():
    src = inspect.getsource(kernel_A.lse_stats_triton)
    banned = [
        "matmul",
        "einsum",
        "softmax",
        "logsumexp",
        "exp(",
        "log(",
        "sum(",
        "max(",
        "mean(",
        "reshape",
        "transpose",
    ]
    lowered = src.lower()
    for key in banned:
        assert key not in lowered, f"wrapper function contains forbidden call token: {key}"


def get_kernel_cache_size():
    return _cache_leaf_count(kernel_A._lse_stats_kernel)



def test_no_recompile_on_D():
    torch.manual_seed(0)
    dims = kernel_A.SUPPORTED_HEAD_DIMS
    B, T = 1, 16
    dtype = torch.float16

    base_keys = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)

    Hq, Hkv = 8, 8
    for D in dims:
        q = torch.randn((B, Hq, T, D), device="cuda", dtype=dtype)
        k = torch.randn((B, Hkv, T, D), device="cuda", dtype=dtype)
        kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=True)

    cache_after = get_kernel_cache_size()
    after_keys = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | A_cache_after_warmup={cache_after}")
    if cache_after != len(base_keys) + len(dims):
        _print_cache_growth("A_cache_after_warmup_mismatch", base_keys, after_keys)
    assert cache_after == len(base_keys) + len(dims), "specialization count does not match supported set"

    for D in dims:
        q = torch.randn((B, Hq, T, D), device="cuda", dtype=dtype)
        k = torch.randn((B, Hkv, T, D), device="cuda", dtype=dtype)
        kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=True)

    cache_after_second = get_kernel_cache_size()
    after_keys_second = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | A_cache_after_second={cache_after_second}")
    if cache_after_second != cache_after:
        _print_cache_growth("A_cache_after_second_growth", after_keys, after_keys_second)
    assert cache_after_second == cache_after, "repeated calls should not increase compiled entries"

    # GQA: no growth is expected only when g is fixed
    gqa_cases = [(8, 4), (16, 8), (8, 2)]
    base_g = gqa_cases[0][0] // gqa_cases[0][1]
    for Hq, Hkv in gqa_cases:
        q = torch.randn((B, Hq, T, dims[0]), device="cuda", dtype=dtype)
        k = torch.randn((B, Hkv, T, dims[0]), device="cuda", dtype=dtype)
        kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=True)
        cur_keys = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)
        cur_cache = len(cur_keys)
        g = Hq // Hkv
        if g == base_g:
            if cur_cache != cache_after_second:
                _print_cache_growth("A_cache_gqa_growth", after_keys_second, cur_keys)
            assert cur_cache == cache_after_second, "no compiled entry increase expected when g is fixed"
        else:
            _print_cache_growth("A_cache_gqa_ratio_changed", after_keys_second, cur_keys)


def test_gqa_ratio_cache():
    torch.manual_seed(42)
    B, T, D = 1, 16, kernel_A.SUPPORTED_HEAD_DIMS[0]
    dtype = torch.float16
    causal = True

    q0 = torch.randn((B, 8, T, D), device="cuda", dtype=dtype)
    k0 = torch.randn((B, 4, T, D), device="cuda", dtype=dtype)
    kernel_A.lse_stats_triton(q0, k0, attn_mask=None, causal=causal)
    keys0 = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)

    q1 = torch.randn((B, 16, T, D), device="cuda", dtype=dtype)
    k1 = torch.randn((B, 8, T, D), device="cuda", dtype=dtype)
    kernel_A.lse_stats_triton(q1, k1, attn_mask=None, causal=causal)
    keys1 = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)
    if len(keys1) != len(keys0):
        _print_cache_growth("A_gqa_ratio_fixed_growth", keys0, keys1)
    assert len(keys1) == len(keys0), "no new compiled entries expected when g is fixed"

    q2 = torch.randn((B, 8, T, D), device="cuda", dtype=dtype)
    k2 = torch.randn((B, 2, T, D), device="cuda", dtype=dtype)
    kernel_A.lse_stats_triton(q2, k2, attn_mask=None, causal=causal)
    keys2 = _get_cache_leaf_keys(kernel_A._lse_stats_kernel)
    _print_cache_growth("A_gqa_ratio_changed", keys1, keys2)


# Numerical correctness tests

def test_correctness():
    torch.manual_seed(1)
    dims = kernel_A.SUPPORTED_HEAD_DIMS
    cases = []

    gqa_pairs = [(8, 8), (8, 4)]
    for D in dims:
        for Hq, Hkv in gqa_pairs:
            cases.append((torch.float16, True, None, 2, Hq, Hkv, 33, D, 0.0))
            cases.append((torch.float16, False, 0.3, 2, Hq, Hkv, 33, D, 0.3))

    # Additional GQA coverage and mask_ratio
    cases.append((torch.float16, True, 0.7, 2, 8, 2, 33, dims[0], 0.7))
    cases.append((torch.float16, False, 0.3, 2, 16, 4, 33, dims[1], 0.3))

    # bf16 coverage
    cases.append((torch.bfloat16, True, None, 2, 8, 8, 33, dims[0], 0.0))
    cases.append((torch.bfloat16, False, 0.3, 2, 8, 4, 33, dims[-1], 0.3))

    # Length not divisible by BLOCK_N
    cases.append((torch.float16, True, None, 1, 8, 4, 127, 64, 0.0))
    cases.append((torch.float16, False, 0.3, 1, 8, 2, 127, 64, 0.3))

    for dtype, causal, mask_ratio, B, Hq, Hkv, T, D, ratio_print in cases:
        q = torch.randn((B, Hq, T, D), device="cuda", dtype=dtype)
        k = torch.randn((B, Hkv, T, D), device="cuda", dtype=dtype)

        if mask_ratio is None:
            mask = None
        else:
            mask = (torch.rand((B, T), device="cuda") > mask_ratio).to(torch.int32)
            mask[:, 0] = 1

        sm_scale = 1.0 / (D ** 0.5)
        ref = reference_lse(q, k, mask, causal, sm_scale)
        out = kernel_A.lse_stats_triton(q, k, attn_mask=mask, causal=causal, sm_scale=sm_scale)

        diff = (out - ref).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        cache_count = get_kernel_cache_size()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(
            f"timestamp={now} | max_abs_err={max_abs_err:.6f} | mean_abs_err={mean_abs_err:.6f} | "
            f"dtype={dtype} causal={causal} has_mask={mask is not None} B={B} Hq={Hq} Hkv={Hkv} T={T} D={D} mask_ratio={ratio_print} "
            f"cache={cache_count}"
        )

        if dtype == torch.float16:
            rtol = 1e-2
            atol = 2e-2
        else:
            rtol = 2e-2
            atol = 3e-2

        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def _sample_gpu_util(stop_event, samples):
    while not stop_event.is_set():
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if vals:
            samples.append(max(vals))
        time.sleep(0.02)


def test_gpu_utilization():
    torch.manual_seed(2)
    B, Hq, Hkv, T, D = 8, 16, 4, 2048, 128
    dtype = torch.float16
    q = torch.randn((B, Hq, T, D), device="cuda", dtype=dtype)
    k = torch.randn((B, Hkv, T, D), device="cuda", dtype=dtype)

    # Warm up compilation
    kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=False)
    torch.cuda.synchronize()

    stop_event = threading.Event()
    samples = []
    t = threading.Thread(target=_sample_gpu_util, args=(stop_event, samples))
    t.start()

    for _ in range(100):
        kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=False)
    torch.cuda.synchronize()

    stop_event.set()
    t.join()

    max_util = max(samples) if samples else 0
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | max_gpu_util={max_util}% | B={B} Hq={Hq} Hkv={Hkv} T={T} D={D}")
    assert max_util >= 90, "GPU utilization did not reach 90%"


def test_long_context_runs():
    torch.manual_seed(3)
    dtype = torch.float16
    B, Hq, Hkv, D = 4, 16, 4, 128
    for T in (4096, 8192, 16384, 32768):
        q = torch.randn((B, Hq, T, D), device="cuda", dtype=dtype)
        k = torch.randn((B, Hkv, T, D), device="cuda", dtype=dtype)
        out = kernel_A.lse_stats_triton(q, k, attn_mask=None, causal=True)
        torch.cuda.synchronize()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"timestamp={now} | long_context_run B={B} Hq={Hq} Hkv={Hkv} T={T} D={D} dtype={dtype}")
        assert out.shape == (B, Hq, T)


if __name__ == "__main__":
    static_audit_wrapper()
    test_no_recompile_on_D()
    test_gqa_ratio_cache()
    test_correctness()
    test_gpu_utilization()
    test_long_context_runs()
