"""Stress-test MLA KV cache write/read for NaN and corruption.

Production NaN is inside MLAAttention.forward_impl — either from
concat_and_cache_mla (KV cache write), forward_mqa (decode kernel
reading the cache), or the W_UK/W_UV absorption BMMs.

This test isolates the KV cache path:
  1. Write known values into the MLA KV cache via concat_and_cache_mla
  2. Read them back and verify no NaN/corruption
  3. Test with FP8 quantized cache (fp8, fp8_e4m3, fp8_ds_mla)
  4. Test concurrent write+read patterns (simulating decode batches)
  5. Test with slot_mapping patterns that stress block boundaries

Runs via torchrun (4 nodes x 4 GPUs = 16 ranks).
"""

import os
import sys
import time

import torch
import torch.distributed as dist

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
BLOCK_SIZE = 16


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[rank={rank} gpu={gpu}] {msg}", file=sys.stderr, flush=True)


def test_cache_write_read_roundtrip(device, dtype):
    """Write values to KV cache and read them back — check for NaN."""
    from vllm import _custom_ops as ops

    log("--- Test 1: KV cache write/read roundtrip ---")
    fails = 0

    for num_tokens in [1, 64, 256, 1024, 4096, 8192]:
        num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

        kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

        # Sequential slot mapping (simple case)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                               dtype=dtype, device=device)

        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping,
            kv_cache_dtype="auto",
            scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        # Read back and compare
        kv_cache_flat = kv_cache.view(-1, HEAD_SIZE)
        readback = kv_cache_flat[:num_tokens]
        readback_c = readback[:, :KV_LORA_RANK]
        readback_pe = readback[:, KV_LORA_RANK:]

        nan_c = readback_c.isnan().sum().item()
        nan_pe = readback_pe.isnan().sum().item()

        # Check exact match (auto dtype = no quantization)
        diff_c = (readback_c - kv_c).abs().max().item()
        diff_pe = (readback_pe - k_pe).abs().max().item()

        ok = nan_c == 0 and nan_pe == 0 and diff_c < 1e-6 and diff_pe < 1e-6
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        log(f"  [{status}] tokens={num_tokens:5d}  nan_c={nan_c} nan_pe={nan_pe} "
            f"diff_c={diff_c:.2e} diff_pe={diff_pe:.2e}")

    return fails


def test_cache_fp8_roundtrip(device, dtype):
    """Write to FP8 KV cache and check for NaN (lossy but no NaN expected)."""
    from vllm import _custom_ops as ops

    log("--- Test 2: FP8 KV cache roundtrip ---")
    fails = 0

    for kv_cache_dtype in ["fp8", "fp8_e4m3"]:
        for num_tokens in [1, 256, 1024, 4096, 8192]:
            num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

            kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
            k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
            slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

            # FP8 cache uses uint8 storage
            kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                                   dtype=torch.uint8, device=device)
            scale = torch.tensor(1.0, dtype=torch.float32, device=device)

            ops.concat_and_cache_mla(
                kv_c, k_pe, kv_cache, slot_mapping,
                kv_cache_dtype=kv_cache_dtype, scale=scale,
            )

            # Check cache for NaN by viewing as fp8
            from vllm.platforms import current_platform
            cache_fp8 = kv_cache.view(current_platform.fp8_dtype())
            nan_count = cache_fp8.isnan().sum().item()
            inf_count = cache_fp8.isinf().sum().item()

            ok = nan_count == 0
            status = "PASS" if ok else "FAIL"
            if not ok:
                fails += 1
            log(f"  [{status}] {kv_cache_dtype} tokens={num_tokens:5d}  "
                f"nan={nan_count} inf={inf_count}")

    return fails


def test_cache_scatter_patterns(device, dtype):
    """Test with non-sequential slot mappings that stress block boundaries."""
    from vllm import _custom_ops as ops

    log("--- Test 3: Scatter slot mapping patterns ---")
    fails = 0
    num_tokens = 4096
    num_blocks = 512  # plenty of space

    patterns = {
        "sequential": torch.arange(num_tokens, dtype=torch.int64, device=device),
        "reversed": torch.arange(num_tokens - 1, -1, -1,
                                 dtype=torch.int64, device=device),
        "strided_2": torch.arange(0, num_tokens * 2, 2,
                                  dtype=torch.int64, device=device),
        "random_perm": torch.randperm(num_blocks * BLOCK_SIZE,
                                      device=device)[:num_tokens].to(torch.int64),
        "block_boundary": torch.tensor(
            [i * BLOCK_SIZE + (BLOCK_SIZE - 1) for i in range(num_tokens)],
            dtype=torch.int64, device=device),  # always last slot in block
    }

    for name, slot_map in patterns.items():
        # Clamp to valid range
        max_slot = num_blocks * BLOCK_SIZE - 1
        slot_map = slot_map.clamp(0, max_slot)

        kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
        kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                               dtype=dtype, device=device)

        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_map,
            kv_cache_dtype="auto",
            scale=torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        # Check written values for NaN
        kv_cache_flat = kv_cache.view(-1, HEAD_SIZE)
        written = kv_cache_flat[slot_map]
        nan_count = written.isnan().sum().item()

        # Verify data integrity
        written_c = written[:, :KV_LORA_RANK]
        written_pe = written[:, KV_LORA_RANK:]
        diff_c = (written_c - kv_c).abs().max().item()
        diff_pe = (written_pe - k_pe).abs().max().item()

        ok = nan_count == 0 and diff_c < 1e-6 and diff_pe < 1e-6
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        log(f"  [{status}] {name:20s}  nan={nan_count} "
            f"diff_c={diff_c:.2e} diff_pe={diff_pe:.2e}")

    return fails


def test_cache_overwrite_stress(device, dtype):
    """Repeatedly write to same cache slots (simulates decode steps)."""
    from vllm import _custom_ops as ops

    log("--- Test 4: Repeated overwrite stress ---")
    fails = 0
    num_tokens = 1024
    num_blocks = 128
    iters = 500

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                           dtype=dtype, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    nan_events = 0

    for it in range(iters):
        kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe = torch.randn(num_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping,
            kv_cache_dtype="auto", scale=scale,
        )

        if kv_cache.isnan().any().item():
            nan_events += 1
            if nan_events <= 3:
                log(f"    NaN at iter {it}")

        if (it + 1) % 100 == 0:
            log(f"    ... {it+1}/{iters}, {nan_events} NaN")

    status = "FAIL" if nan_events else "PASS"
    if nan_events:
        fails += 1
    log(f"  [{status}] {nan_events}/{iters} NaN events")
    return fails


def test_cache_input_extremes(device, dtype):
    """Test cache write with extreme input values."""
    from vllm import _custom_ops as ops

    log("--- Test 5: Extreme input values ---")
    fails = 0
    num_tokens = 1024
    num_blocks = 128
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    cases = {
        "zeros": (torch.zeros, torch.zeros),
        "ones": (torch.ones, torch.ones),
        "large_1e3": (
            lambda *a, **kw: torch.randn(*a, **kw) * 1e3,
            lambda *a, **kw: torch.randn(*a, **kw) * 1e3,
        ),
        "tiny_1e-6": (
            lambda *a, **kw: torch.randn(*a, **kw) * 1e-6,
            lambda *a, **kw: torch.randn(*a, **kw) * 1e-6,
        ),
        "max_bf16": (
            lambda *a, **kw: torch.full(a, 65504.0, **kw),
            lambda *a, **kw: torch.full(a, 65504.0, **kw),
        ),
        "sparse_outliers": (
            lambda n, d, **kw: _sparse_outlier(n, d, **kw),
            lambda n, d, **kw: _sparse_outlier(n, d, **kw),
        ),
    }

    for name, (gen_c, gen_pe) in cases.items():
        kv_c = gen_c(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
        k_pe = gen_pe(num_tokens, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)
        kv_cache = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                               dtype=dtype, device=device)

        ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping,
            kv_cache_dtype="auto", scale=scale,
        )

        nan_count = kv_cache.isnan().sum().item()
        ok = nan_count == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        log(f"  [{status}] {name:20s}  nan={nan_count}")

    return fails


def _sparse_outlier(n, d, dtype, device):
    x = torch.randn(n, d, dtype=dtype, device=device) * 0.01
    mask = torch.rand(n, d, device=device) < 0.01
    x[mask] = torch.randn(mask.sum().item(), dtype=dtype, device=device) * 100
    return x


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16  # production uses bf16 activations

    log(f"Device: {torch.cuda.get_device_name(device)}")

    total_fails = 0

    log("")
    log("=" * 72)
    total_fails += test_cache_write_read_roundtrip(device, dtype)

    log("")
    log("=" * 72)
    total_fails += test_cache_fp8_roundtrip(device, dtype)

    log("")
    log("=" * 72)
    total_fails += test_cache_scatter_patterns(device, dtype)

    log("")
    log("=" * 72)
    total_fails += test_cache_overwrite_stress(device, dtype)

    log("")
    log("=" * 72)
    total_fails += test_cache_input_extremes(device, dtype)

    log("")
    log("=" * 72)
    log(f"TOTAL FAILURES: {total_fails}")
    log("RESULT: " + ("FAIL" if total_fails else "PASS"))
    log("=" * 72)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    main()
