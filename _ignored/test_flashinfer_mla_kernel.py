"""Stress-test the TRT-LLM SM100 FMHA kernel for NaN via FlashInfer MLA.

This test targets the EXACT kernel producing NaN in production:
  fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseP64
  VarSeqQ64Kv128Persistent2CtaKeepsAbForGen

The kernel is dispatched by flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla()
when the kv_cache tensor dtype is torch.float8_e4m3fn.

Production finding: NaN appears at layer 1 with ALL inputs clean:
  kv_cache_upd=0  W_UK_bmm=0  fwd_mqa=NaN  v_up_proj=NaN(propagated)

The NaN count matches: fwd_mqa=65536 = 1 row * 128 heads * 512 kv_lora_rank.
With 5 affected rows: 327680 = 5 * 128 * 512.

Key differences from the BF16 path:
  - KV cache tensor is torch.float8_e4m3fn (dispatches to QkvE4m3 kernel)
  - bmm1_scale = q_scale * k_scale * attn_scale (combines FP8 dequant with softmax scale)
  - bmm2_scale = v_scale (FP8 value dequant scale)

Runs via torchrun (4 nodes x 4 GPUs = 16 ranks).
"""

import os
import sys
import time

import torch
import torch.distributed as dist

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
NUM_HEADS = 128
BLOCK_SIZE = 64

# Attention scale used in production
ATTN_SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[rank={rank} gpu={gpu}] {msg}", file=sys.stderr, flush=True)


def _check_flashinfer_mla():
    """Return True if FlashInfer MLA is available."""
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        return True
    except ImportError:
        try:
            from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
            return True
        except ImportError:
            return False


def _run_flashinfer_mla_decode(q, kv_cache, block_tables, seq_lens,
                                num_heads, device,
                                bmm1_scale=None, bmm2_scale=None):
    """Run the FlashInfer MLA decode kernel directly.

    When kv_cache.dtype is torch.float8_e4m3fn, this dispatches to:
      fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512...
    which is the exact kernel producing NaN in production.
    """
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    bs = len(seq_lens)

    if bmm1_scale is None:
        bmm1_scale = ATTN_SCALE
    if bmm2_scale is None:
        bmm2_scale = 1.0

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    q_reshaped = q.view(bs, 1, num_heads, HEAD_SIZE)
    kv_cache_4d = kv_cache.unsqueeze(1)

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q_reshaped,
        kv_cache=kv_cache_4d,
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max(seq_lens).item() if isinstance(seq_lens, torch.Tensor) else max(seq_lens),
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    return o.view(bs, num_heads, KV_LORA_RANK)


def _make_fp8_kv_cache_and_tables(bs, ctx_len, num_blocks, device,
                                    cache_fill="randn", cache_scale=1.0):
    """Create FP8 E4M3 KV cache and block tables.

    This is the critical difference from BF16 tests: the KV cache tensor
    is torch.float8_e4m3fn, which causes the kernel to dispatch to the
    QkvE4m3 variant that's producing NaN in production.
    """
    # Generate in BF16 first, then quantize to FP8
    kv_bf16 = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                           dtype=torch.bfloat16, device=device)

    blocks_per_seq = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = blocks_per_seq
    block_tables = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)

    next_block = 1
    for i in range(bs):
        for j in range(blocks_per_seq):
            if next_block >= num_blocks:
                break
            block_tables[i, j] = next_block

            fill_len = min(BLOCK_SIZE, ctx_len - j * BLOCK_SIZE)
            if fill_len <= 0:
                break

            if cache_fill == "randn":
                kv_bf16[next_block, :fill_len] = (
                    torch.randn(fill_len, HEAD_SIZE, dtype=torch.bfloat16, device=device)
                    * cache_scale
                )
            elif cache_fill == "ones":
                kv_bf16[next_block, :fill_len] = 1.0
            elif cache_fill == "large":
                kv_bf16[next_block, :fill_len] = (
                    torch.randn(fill_len, HEAD_SIZE, dtype=torch.bfloat16, device=device)
                    * 100.0
                )
            elif cache_fill == "sparse":
                data = torch.zeros(fill_len, HEAD_SIZE, dtype=torch.bfloat16, device=device)
                mask = torch.rand(fill_len, HEAD_SIZE, device=device) < 0.01
                data[mask] = (
                    torch.randn(mask.sum().item(), dtype=torch.bfloat16, device=device) * 50
                )
                kv_bf16[next_block, :fill_len] = data
            elif cache_fill == "fp8_grid":
                # Fill with values that are exactly representable in FP8 E4M3
                raw = torch.randn(fill_len, HEAD_SIZE, dtype=torch.float32, device=device)
                raw = raw * cache_scale
                kv_bf16[next_block, :fill_len] = raw.to(torch.float8_e4m3fn).to(torch.bfloat16)

            next_block += 1

    # Quantize entire cache to FP8 E4M3
    kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

    seq_lens = torch.full((bs,), ctx_len, dtype=torch.int32, device=device)
    return kv_fp8, block_tables, seq_lens


def _compute_production_scales(q, kv_c_bf16):
    """Compute bmm1_scale and bmm2_scale the same way production does.

    Production computes:
      q_scale = abs(q).max() / q_range   (q_range = FP8_E4M3_MAX)
      k_scale = abs(kv_c).max() / k_range
      v_scale = abs(kv_c).max() / v_range
      bmm1_scale = q_scale * k_scale * attn_scale
      bmm2_scale = v_scale
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    q_abs_max = q.abs().max().item()
    kv_abs_max = kv_c_bf16.abs().max().item()

    q_scale = q_abs_max / fp8_max if q_abs_max > 0 else 1.0
    k_scale = kv_abs_max / fp8_max if kv_abs_max > 0 else 1.0
    v_scale = kv_abs_max / fp8_max if kv_abs_max > 0 else 1.0

    bmm1_scale = q_scale * k_scale * ATTN_SCALE
    bmm2_scale = v_scale

    return bmm1_scale, bmm2_scale


# ---------------------------------------------------------------------------
# Tests targeting the FP8 E4M3 kernel variant (QkvE4m3)
# ---------------------------------------------------------------------------

def test_fp8_batch_sweep(device):
    """FP8 E4M3 kernel at various batch sizes — the production kernel."""
    log("--- Test 1: FP8 E4M3 batch size sweep (ctx=256) ---")
    fails = 0
    ctx = 256

    for bs in [1, 32, 128, 256, 512, 1024, 2048, 4096]:
        blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * blocks_per_seq + 2

        kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
            bs, ctx, num_blocks, device)

        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)

        # Use production-like dequant scales
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ic = out.isinf().sum().item()
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            ok = nc == 0 and ic == 0
            if not ok:
                fails += 1
            log(f"  [{'PASS' if ok else 'FAIL'}] bs={bs:5d}  "
                f"nan={nc} inf={ic} abs_max={am:.4f} "
                f"bmm1={bmm1:.6e} bmm2={bmm2:.6e}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] bs={bs:5d}  {e}")

    return fails


def test_fp8_context_sweep(device):
    """FP8 E4M3 kernel at various context lengths."""
    log("--- Test 2: FP8 E4M3 context length sweep (bs=512) ---")
    fails = 0
    bs = 512

    for ctx in [16, 64, 256, 512, 1024, 2048, 4096]:
        blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * blocks_per_seq + 2

        kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
            bs, ctx, num_blocks, device)

        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] ctx={ctx:5d}  "
                f"nan={nc} abs_max={am:.4f}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] ctx={ctx:5d}  {e}")

    return fails


def test_fp8_scale_sweep(device):
    """Test the FP8 kernel with various bmm1_scale/bmm2_scale values.

    In production, these scales depend on dynamic quantization of Q and KV.
    Values can vary widely depending on activation magnitudes.
    """
    log("--- Test 3: FP8 scale sweep ---")
    fails = 0
    bs = 512
    ctx = 256
    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * blocks_per_seq + 2

    kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
        bs, ctx, num_blocks, device)

    q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                     dtype=torch.bfloat16, device=device)

    # Test various scale combinations
    # Production bmm1_scale = q_scale * k_scale * attn_scale
    # where q_scale, k_scale ~ abs_max/448.0, attn_scale ~ 1/sqrt(192) ~ 0.072
    # Typical: q_scale ~ 0.01-0.1, k_scale ~ 0.01-0.1
    # So bmm1_scale ~ 1e-5 to 1e-3
    scale_cases = [
        ("production_typical", 1e-4, 0.01),
        ("attn_scale_only", ATTN_SCALE, 1.0),
        ("very_small_bmm1", 1e-8, 0.001),
        ("very_large_bmm1", 1.0, 1.0),
        ("large_bmm2", 1e-4, 10.0),
        ("tiny_bmm2", 1e-4, 1e-6),
        ("both_large", 1.0, 10.0),
        ("production_computed", None, None),  # computed from data
    ]

    for name, bmm1, bmm2 in scale_cases:
        if bmm1 is None:
            bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] {name:25s}  "
                f"bmm1={bmm1:.6e} bmm2={bmm2:.6e}  "
                f"nan={nc} abs_max={am:.4f}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {name}  {e}")

    return fails


def test_fp8_cache_fill_patterns(device):
    """FP8 kernel with different KV cache fill patterns."""
    log("--- Test 4: FP8 KV cache fill patterns ---")
    fails = 0
    bs = 512
    ctx = 256

    for fill, scale in [
        ("randn", 1.0),
        ("randn", 0.01),
        ("randn", 100.0),
        ("ones", 1.0),
        ("large", 1.0),
        ("sparse", 1.0),
        ("fp8_grid", 1.0),
        ("fp8_grid", 0.01),
        ("fp8_grid", 100.0),
    ]:
        blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * blocks_per_seq + 2

        kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
            bs, ctx, num_blocks, device,
            cache_fill=fill, cache_scale=scale)

        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            name = f"{fill}(s={scale})"
            log(f"  [{'PASS' if ok else 'FAIL'}] {name:20s}  "
                f"nan={nc} abs_max={am:.4f}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {fill}(s={scale})  {e}")

    return fails


def test_fp8_stress(device):
    """Repeated FP8 kernel calls to catch intermittent NaN."""
    log("--- Test 5: FP8 stress test (bs=2048, ctx=256, 500 iters) ---")
    bs = 2048
    ctx = 256
    iters = 500

    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * blocks_per_seq + 2

    # Create a fixed FP8 KV cache
    kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
        bs, ctx, num_blocks, device)

    nan_events = 0
    for it in range(iters):
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            if out.isnan().any().item():
                nan_events += 1
                if nan_events <= 5:
                    nc = out.isnan().sum().item()
                    log(f"    NaN at iter {it}, count={nc}")
        except Exception as e:
            if it == 0:
                log(f"    KERNEL ERROR (aborting stress test): {e}")
                return 1
            nan_events += 1

        if (it + 1) % 100 == 0:
            log(f"    ... {it+1}/{iters}, {nan_events} NaN")

    status = "FAIL" if nan_events else "PASS"
    log(f"  [{status}] {nan_events}/{iters} NaN events")
    return 1 if nan_events > 0 else 0


def test_fp8_mixed_seq_lens(device):
    """FP8 kernel with heterogeneous sequence lengths."""
    log("--- Test 6: FP8 mixed sequence lengths ---")
    fails = 0

    import random
    random.seed(42)
    cases = {
        "all_short": [32] * 512,
        "all_long": [2048] * 128,
        "mixed_uniform": [random.randint(16, 4096) for _ in range(256)],
        "mostly_short_few_long": [32] * 480 + [4096] * 32,
        "power_of_2": [2**i for i in range(4, 13)] * 20,
    }

    for name, seq_lens_list in cases.items():
        bs = len(seq_lens_list)
        max_ctx = max(seq_lens_list)
        num_blocks = sum((s + BLOCK_SIZE - 1) // BLOCK_SIZE
                         for s in seq_lens_list) + 2

        # Build FP8 KV cache
        kv_bf16 = torch.randn(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                                dtype=torch.bfloat16, device=device)
        kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

        max_blocks_per_seq = (max_ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_tables = torch.zeros(bs, max_blocks_per_seq,
                                    dtype=torch.int32, device=device)

        next_block = 1
        for i in range(bs):
            bps = (seq_lens_list[i] + BLOCK_SIZE - 1) // BLOCK_SIZE
            for j in range(bps):
                if next_block < num_blocks:
                    block_tables[i, j] = next_block
                    next_block += 1

        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} bs={bs:4d}  "
                f"nan={nc} abs_max={am:.4f}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {name}  {e}")

    return fails


def test_bf16_vs_fp8_comparison(device):
    """Run the SAME inputs through BF16 and FP8 kernels and compare.

    This isolates whether the FP8 kernel variant specifically produces NaN
    while the BF16 variant does not.
    """
    log("--- Test 7: BF16 vs FP8 kernel comparison ---")
    fails = 0
    bs = 1024
    ctx = 256

    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * blocks_per_seq + 2

    for trial in range(10):
        # Generate KV cache in BF16
        kv_bf16 = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                               dtype=torch.bfloat16, device=device)

        block_tables = torch.zeros(bs, blocks_per_seq, dtype=torch.int32, device=device)
        next_block = 1
        for i in range(bs):
            for j in range(blocks_per_seq):
                if next_block < num_blocks:
                    block_tables[i, j] = next_block
                    kv_bf16[next_block, :BLOCK_SIZE] = torch.randn(
                        BLOCK_SIZE, HEAD_SIZE, dtype=torch.bfloat16, device=device)
                    next_block += 1

        seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)

        # FP8 version (quantize cache)
        kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)
        bmm1_fp8, bmm2_fp8 = _compute_production_scales(q, kv_bf16)

        # BF16 version
        try:
            out_bf16 = _run_flashinfer_mla_decode(
                q, kv_bf16, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=ATTN_SCALE, bmm2_scale=1.0)
            bf16_nan = out_bf16.isnan().sum().item()
        except Exception as e:
            bf16_nan = -1
            log(f"    BF16 error trial {trial}: {e}")

        try:
            out_fp8 = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1_fp8, bmm2_scale=bmm2_fp8)
            fp8_nan = out_fp8.isnan().sum().item()
        except Exception as e:
            fp8_nan = -1
            log(f"    FP8 error trial {trial}: {e}")

        ok = bf16_nan == 0 and fp8_nan == 0
        if not ok:
            fails += 1
        log(f"  [{'PASS' if ok else 'FAIL'}] trial={trial}  "
            f"bf16_nan={bf16_nan} fp8_nan={fp8_nan}")

    return fails


def test_fp8_extreme_queries(device):
    """FP8 kernel with extreme query distributions.

    In production, queries come from post-RMSNorm + RoPE + W_UK BMM.
    Some distributions can cause the FMHA softmax to overflow in FP8 mode.
    """
    log("--- Test 8: FP8 extreme query distributions ---")
    fails = 0
    bs = 512
    ctx = 256
    blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * blocks_per_seq + 2

    kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
        bs, ctx, num_blocks, device)

    qdim = HEAD_SIZE

    def _post_rmsnorm(bs, num_heads, dim, device):
        x = torch.randn(bs, num_heads, dim, dtype=torch.float32, device=device)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        return (x / rms).to(torch.bfloat16)

    distributions = {
        "normal_std1": lambda: torch.randn(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device),
        "normal_std0.01": lambda: torch.randn(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device) * 0.01,
        "normal_std100": lambda: torch.randn(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device) * 100,
        "uniform_-1_1": lambda: torch.rand(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device) * 2 - 1,
        "post_rmsnorm": lambda: _post_rmsnorm(bs, NUM_HEADS, qdim, device),
        "all_positive": lambda: torch.randn(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device).abs(),
        "near_fp8_max": lambda: torch.randn(bs, NUM_HEADS, qdim, dtype=torch.bfloat16, device=device) * 400,
        "mixed_magnitude": lambda: _mixed_magnitude(bs, NUM_HEADS, qdim, device),
    }

    for name, gen in distributions.items():
        q = gen()
        bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] {name:20s}  "
                f"nan={nc} abs_max={am:.4f}  "
                f"bmm1={bmm1:.6e} bmm2={bmm2:.6e}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {name}  {e}")

    return fails


def test_fp8_large_batch_stress(device):
    """Stress test at production-scale batch sizes with FP8 KV cache.

    Production runs decode with bs up to 1024 per GPU. NaN was observed
    at layer 1, first forward pass, affecting specific rows only.
    This test hammers the kernel at production scale.
    """
    log("--- Test 9: FP8 large batch stress (500 iters, varying bs) ---")
    total_nan = 0
    iters = 500

    for bs in [64, 1024, 2048, 4096]:
        ctx = 256
        blocks_per_seq = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * blocks_per_seq + 2

        kv_fp8, block_tables, seq_lens = _make_fp8_kv_cache_and_tables(
            bs, ctx, num_blocks, device)

        nan_events = 0
        t0 = time.time()
        for it in range(iters):
            q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                             dtype=torch.bfloat16, device=device)
            bmm1, bmm2 = _compute_production_scales(q, kv_fp8.to(torch.bfloat16))

            try:
                out = _run_flashinfer_mla_decode(
                    q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                    bmm1_scale=bmm1, bmm2_scale=bmm2)
                if out.isnan().any().item():
                    nan_events += 1
                    if nan_events <= 3:
                        nc = out.isnan().sum().item()
                        # Check how many rows have NaN
                        nan_rows = out.isnan().any(dim=-1).any(dim=-1).sum().item()
                        log(f"    NaN iter={it} count={nc} nan_rows={nan_rows}")
            except Exception as e:
                if it == 0:
                    log(f"    KERNEL ERROR (aborting bs={bs}): {e}")
                    total_nan += 1
                    break
                nan_events += 1

            if (it + 1) % 100 == 0:
                elapsed = time.time() - t0
                log(f"    bs={bs} ... {it+1}/{iters}, {nan_events} NaN, {elapsed:.1f}s")

        status = "FAIL" if nan_events else "PASS"
        log(f"  [{status}] bs={bs}: {nan_events}/{iters} NaN events")
        total_nan += nan_events

    return 1 if total_nan > 0 else 0


def test_fp8_block_size_sweep(device):
    """Test FP8 kernel with different block sizes (32 and 64)."""
    log("--- Test 10: FP8 block size sweep ---")
    fails = 0
    bs = 512
    ctx = 256

    for block_size in [32, 64]:
        blocks_per_seq = (ctx + block_size - 1) // block_size
        num_blocks = bs * blocks_per_seq + 2

        kv_bf16 = torch.randn(num_blocks, block_size, HEAD_SIZE,
                                dtype=torch.bfloat16, device=device)
        kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

        block_tables = torch.zeros(bs, blocks_per_seq, dtype=torch.int32, device=device)
        next_block = 1
        for i in range(bs):
            for j in range(blocks_per_seq):
                if next_block < num_blocks:
                    block_tables[i, j] = next_block
                    next_block += 1

        seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _compute_production_scales(q, kv_bf16)

        try:
            out = _run_flashinfer_mla_decode(
                q, kv_fp8, block_tables, seq_lens, NUM_HEADS, device,
                bmm1_scale=bmm1, bmm2_scale=bmm2)
            nc = out.isnan().sum().item()
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] block_size={block_size}  "
                f"nan={nc} abs_max={am:.4f}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] block_size={block_size}  {e}")

    return fails


def _mixed_magnitude(bs, num_heads, dim, device):
    """Create queries with mixed magnitudes — some heads large, some small."""
    q = torch.randn(bs, num_heads, dim, dtype=torch.bfloat16, device=device)
    # Make some heads have very large values
    q[:, :32, :] *= 100
    q[:, 32:64, :] *= 0.01
    return q


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    log(f"Device: {torch.cuda.get_device_name(device)}")
    log(f"SM: {torch.cuda.get_device_properties(device).major}."
        f"{torch.cuda.get_device_properties(device).minor}")

    if not _check_flashinfer_mla():
        log("SKIP: FlashInfer MLA not available (no trtllm_batch_decode_with_kv_cache_mla)")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)
    log("FlashInfer MLA kernel available")

    # Verify FP8 support
    try:
        test_fp8 = torch.zeros(1, dtype=torch.float8_e4m3fn, device=device)
        log(f"FP8 E4M3 support: OK (max={torch.finfo(torch.float8_e4m3fn).max})")
    except Exception as e:
        log(f"ERROR: FP8 E4M3 not supported: {e}")
        sys.exit(1)

    total_fails = 0

    log("")
    log("=" * 72)
    log("TARGET KERNEL: fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512...")
    log("  (dispatched when kv_cache.dtype == torch.float8_e4m3fn)")
    log("=" * 72)

    log("")
    log("=" * 72)
    total_fails += test_fp8_batch_sweep(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_context_sweep(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_scale_sweep(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_cache_fill_patterns(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_stress(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_mixed_seq_lens(device)

    log("")
    log("=" * 72)
    total_fails += test_bf16_vs_fp8_comparison(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_extreme_queries(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_large_batch_stress(device)

    log("")
    log("=" * 72)
    total_fails += test_fp8_block_size_sweep(device)

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
