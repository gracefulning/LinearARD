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
from kernels.attention_KL import kernel_B


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
def reference_kl(qs, ks, lse_s, qt, kt, lse_t, attn_mask, causal, sm_scale_s, sm_scale_t):
    qs_f = qs.float()
    ks_f = ks.float()
    qt_f = qt.float()
    kt_f = kt.float()

    B, Hq, T, _ = qs.shape
    Hkv_s = ks.shape[1]
    Hkv_t = kt.shape[1]
    g_s = Hq // Hkv_s
    g_t = Hq // Hkv_t

    map_s = torch.arange(Hq, device=qs.device) // g_s
    map_t = torch.arange(Hq, device=qs.device) // g_t
    ks_map = ks_f[:, map_s, :, :]
    kt_map = kt_f[:, map_t, :, :]

    ss = torch.matmul(qs_f, ks_map.transpose(-1, -2)) * sm_scale_s
    st = torch.matmul(qt_f, kt_map.transpose(-1, -2)) * sm_scale_t

    if attn_mask is not None:
        mask_bool = attn_mask.to(torch.bool)
        valid_k = mask_bool[:, None, None, :]
        valid_q = mask_bool[:, None, :]
    else:
        valid_k = torch.ones((B, 1, 1, T), device=qs.device, dtype=torch.bool)
        valid_q = torch.ones((B, 1, T), device=qs.device, dtype=torch.bool)

    if causal:
        causal_mask = torch.tril(torch.ones((T, T), device=qs.device, dtype=torch.bool))
        valid = valid_k & causal_mask[None, None, :, :]
    else:
        valid = valid_k

    logps = ss - lse_s[..., None]
    logpt = st - lse_t[..., None]
    logpt = torch.where(valid, logpt, torch.zeros_like(logpt))
    logps = torch.where(valid, logps, torch.zeros_like(logps))
    pt = torch.where(valid, torch.exp(logpt), torch.zeros_like(logpt))

    term = pt * (logpt - logps)
    term = torch.where(valid_q[..., None], term, torch.zeros_like(term))

    return term.sum().to(torch.float32)



def static_audit_wrapper():
    src = inspect.getsource(kernel_B.kl_forward_triton)
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
    return _cache_leaf_count(kernel_B._kl_forward_kernel)



def test_no_recompile_on_D():
    torch.manual_seed(0)
    dims = kernel_B.SUPPORTED_HEAD_DIMS
    combos = [(ds, dt) for ds in dims for dt in dims]
    B, T = 1, 16
    dtype = torch.float16

    base_keys = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)

    Hq, Hkv_s, Hkv_t = 8, 8, 8
    for Ds, Dt in combos:
        qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
        ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
        qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
        kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)
        lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=None, causal=True)
        lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=None, causal=True)
        kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=True)

    cache_after = get_kernel_cache_size()
    after_keys = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | B_cache_after_warmup={cache_after}")
    if cache_after != len(base_keys) + len(combos):
        _print_cache_growth("B_cache_after_warmup_mismatch", base_keys, after_keys)
    assert cache_after == len(base_keys) + len(combos), "specialization count does not match supported set"

    for Ds, Dt in combos:
        qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
        ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
        qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
        kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)
        lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=None, causal=True)
        lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=None, causal=True)
        kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=True)

    cache_after_second = get_kernel_cache_size()
    after_keys_second = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | B_cache_after_second={cache_after_second}")
    if cache_after_second != cache_after:
        _print_cache_growth("B_cache_after_second_growth", after_keys, after_keys_second)
    assert cache_after_second == cache_after, "repeated calls should not increase compiled entries"

    # GQA: no growth is expected only when g_s/g_t are fixed
    gqa_cases = [(8, 4, 2), (16, 8, 4), (8, 2, 2)]
    Ds, Dt = dims[0], dims[-1]
    base_gs = gqa_cases[0][0] // gqa_cases[0][1]
    base_gt = gqa_cases[0][0] // gqa_cases[0][2]
    for Hq, Hkv_s, Hkv_t in gqa_cases:
        qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
        ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
        qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
        kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)
        lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=None, causal=True)
        lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=None, causal=True)
        kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=True)
        cur_keys = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)
        cur_cache = len(cur_keys)
        g_s = Hq // Hkv_s
        g_t = Hq // Hkv_t
        if g_s == base_gs and g_t == base_gt:
            if cur_cache != cache_after_second:
                _print_cache_growth("B_cache_gqa_growth", after_keys_second, cur_keys)
            assert cur_cache == cache_after_second, "no compiled entry increase expected when g_s/g_t are fixed"
        else:
            _print_cache_growth("B_cache_gqa_ratio_changed", after_keys_second, cur_keys)


def test_gqa_ratio_cache():
    torch.manual_seed(42)
    B, T = 1, 16
    Ds, Dt = kernel_B.SUPPORTED_HEAD_DIMS[-1], kernel_B.SUPPORTED_HEAD_DIMS[2]
    dtype = torch.float16
    causal = True

    qs0 = torch.randn((B, 8, T, Ds), device="cuda", dtype=dtype)
    ks0 = torch.randn((B, 4, T, Ds), device="cuda", dtype=dtype)
    qt0 = torch.randn((B, 8, T, Dt), device="cuda", dtype=dtype)
    kt0 = torch.randn((B, 2, T, Dt), device="cuda", dtype=dtype)
    lse_s0 = kernel_A.lse_stats_triton(qs0, ks0, attn_mask=None, causal=causal)
    lse_t0 = kernel_A.lse_stats_triton(qt0, kt0, attn_mask=None, causal=causal)
    kernel_B.kl_forward_triton(qs0, ks0, lse_s0, qt0, kt0, lse_t0, attn_mask=None, causal=causal)
    keys0 = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)

    qs1 = torch.randn((B, 16, T, Ds), device="cuda", dtype=dtype)
    ks1 = torch.randn((B, 8, T, Ds), device="cuda", dtype=dtype)
    qt1 = torch.randn((B, 16, T, Dt), device="cuda", dtype=dtype)
    kt1 = torch.randn((B, 4, T, Dt), device="cuda", dtype=dtype)
    lse_s1 = kernel_A.lse_stats_triton(qs1, ks1, attn_mask=None, causal=causal)
    lse_t1 = kernel_A.lse_stats_triton(qt1, kt1, attn_mask=None, causal=causal)
    kernel_B.kl_forward_triton(qs1, ks1, lse_s1, qt1, kt1, lse_t1, attn_mask=None, causal=causal)
    keys1 = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)
    if len(keys1) != len(keys0):
        _print_cache_growth("B_gqa_ratio_fixed_growth", keys0, keys1)
    assert len(keys1) == len(keys0), "no new compiled entries expected when g_s/g_t are fixed"

    qs2 = torch.randn((B, 8, T, Ds), device="cuda", dtype=dtype)
    ks2 = torch.randn((B, 2, T, Ds), device="cuda", dtype=dtype)
    qt2 = torch.randn((B, 8, T, Dt), device="cuda", dtype=dtype)
    kt2 = torch.randn((B, 2, T, Dt), device="cuda", dtype=dtype)
    lse_s2 = kernel_A.lse_stats_triton(qs2, ks2, attn_mask=None, causal=causal)
    lse_t2 = kernel_A.lse_stats_triton(qt2, kt2, attn_mask=None, causal=causal)
    kernel_B.kl_forward_triton(qs2, ks2, lse_s2, qt2, kt2, lse_t2, attn_mask=None, causal=causal)
    keys2 = _get_cache_leaf_keys(kernel_B._kl_forward_kernel)
    _print_cache_growth("B_gqa_ratio_changed", keys1, keys2)


# Numerical correctness tests

def test_correctness():
    torch.manual_seed(1)
    dims = kernel_B.SUPPORTED_HEAD_DIMS
    cases = []

    # Full 25 combinations: use smaller shapes
    for Ds in dims:
        for Dt in dims:
            cases.append((torch.float16, True, None, 1, 8, 8, 8, 16, Ds, Dt, 0.0))

    # Additional coverage for causal=False and padding + GQA
    cases.append((torch.float16, False, 0.3, 2, 8, 4, 2, 33, 8, 128, 0.3))
    cases.append((torch.float16, True, 0.7, 2, 8, 2, 4, 33, 128, 8, 0.7))
    cases.append((torch.float16, False, 0.3, 2, 16, 4, 4, 33, 16, 32, 0.3))

    # bf16 coverage
    cases.append((torch.bfloat16, True, None, 2, 8, 8, 8, 33, 64, 64, 0.0))

    # Length not divisible by BLOCK_N
    cases.append((torch.float16, False, 0.3, 1, 8, 4, 2, 127, 64, 128, 0.3))

    for dtype, causal, mask_ratio, B, Hq, Hkv_s, Hkv_t, T, Ds, Dt, ratio_print in cases:
        qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
        ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
        qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
        kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)

        if mask_ratio is None:
            mask = None
        else:
            mask = (torch.rand((B, T), device="cuda") > mask_ratio).to(torch.int32)
            mask[:, 0] = 1

        sm_scale_s = 1.0 / (Ds ** 0.5)
        sm_scale_t = 1.0 / (Dt ** 0.5)

        lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=mask, causal=causal, sm_scale=sm_scale_s)
        lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=mask, causal=causal, sm_scale=sm_scale_t)

        ref = reference_kl(qs, ks, lse_s, qt, kt, lse_t, mask, causal, sm_scale_s, sm_scale_t)
        out = kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=mask, causal=causal, sm_scale_s=sm_scale_s, sm_scale_t=sm_scale_t)

        diff = (out - ref).abs()
        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        cache_count = get_kernel_cache_size()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(
            f"timestamp={now} | max_abs_err={max_abs_err:.6f} | mean_abs_err={mean_abs_err:.6f} | "
            f"dtype={dtype} causal={causal} has_mask={mask is not None} B={B} Hq={Hq} Hkv_s={Hkv_s} Hkv_t={Hkv_t} T={T} Ds={Ds} Dt={Dt} mask_ratio={ratio_print} "
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
    B, Hq, Hkv_s, Hkv_t, T, Ds, Dt = 8, 16, 4, 2, 2048, 128, 128
    dtype = torch.float16
    qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
    ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
    qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
    kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)
    lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=None, causal=False)
    lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=None, causal=False)

    # Warm up compilation
    kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=False)
    torch.cuda.synchronize()

    stop_event = threading.Event()
    samples = []
    t = threading.Thread(target=_sample_gpu_util, args=(stop_event, samples))
    t.start()

    for _ in range(100):
        kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=False)
    torch.cuda.synchronize()

    stop_event.set()
    t.join()

    max_util = max(samples) if samples else 0
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"timestamp={now} | max_gpu_util={max_util}% | B={B} Hq={Hq} Hkv_s={Hkv_s} Hkv_t={Hkv_t} T={T} Ds={Ds} Dt={Dt}")
    assert max_util >= 90, "GPU utilization did not reach 90%"


def test_long_context_runs():
    torch.manual_seed(3)
    dtype = torch.float16
    B, Hq, Hkv_s, Hkv_t, Ds, Dt = 4, 16, 4, 4, 128, 128
    for T in (4096, 8192, 16384, 32768):
        qs = torch.randn((B, Hq, T, Ds), device="cuda", dtype=dtype)
        ks = torch.randn((B, Hkv_s, T, Ds), device="cuda", dtype=dtype)
        qt = torch.randn((B, Hq, T, Dt), device="cuda", dtype=dtype)
        kt = torch.randn((B, Hkv_t, T, Dt), device="cuda", dtype=dtype)
        lse_s = kernel_A.lse_stats_triton(qs, ks, attn_mask=None, causal=True)
        lse_t = kernel_A.lse_stats_triton(qt, kt, attn_mask=None, causal=True)
        out = kernel_B.kl_forward_triton(qs, ks, lse_s, qt, kt, lse_t, attn_mask=None, causal=True)
        torch.cuda.synchronize()
        now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"timestamp={now} | long_context_run B={B} Hq={Hq} Hkv_s={Hkv_s} Hkv_t={Hkv_t} T={T} Ds={Ds} Dt={Dt} dtype={dtype}")
        assert out.numel() == 1


if __name__ == "__main__":
    static_audit_wrapper()
    test_no_recompile_on_D()
    test_gqa_ratio_cache()
    test_correctness()
    test_gpu_utilization()
    test_long_context_runs()
