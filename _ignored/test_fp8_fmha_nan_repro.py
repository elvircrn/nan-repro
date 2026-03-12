"""Minimal reproducer for TRT-LLM SM100 FMHA NaN with FP8 E4M3 KV cache.

Targets: fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseP64
         VarSeqQ64Kv128Persistent2CtaKeepsAbForGen

Production bug characteristics:
  - NaN appears at layer 1, first decode step after NIXL KV transfer
  - Affects specific rows only (1 out of 1024, or 5 out of 1024)
  - All inputs to fwd_mqa are clean (no NaN in Q, KV cache, block tables)
  - NaN count = affected_rows * 128 heads * 512 kv_lora_rank
  - Every GPU shows NaN (not a single-GPU issue)
  - BF16 kernel variant may not reproduce (different CUDA kernel)

This test reproduces the exact production data flow:
  1. Generate KV cache as BF16 (simulating post-embedding/post-LN values)
  2. Dynamic FP8 quantization (q_scale, k_scale, v_scale from abs max)
  3. Store KV cache as FP8 E4M3 (simulating concat_and_cache_mla)
  4. Call trtllm_batch_decode_with_kv_cache_mla with FP8 cache
  5. Check output for NaN

Hypotheses tested:
  A. Kernel bug with specific FP8 value patterns (denormals, near-max)
  B. Kernel bug with specific scale combinations
  C. Kernel bug with specific batch size / seq_len combinations
  D. Kernel bug with block table layout (non-contiguous blocks)
  E. Race condition in persistent 2-CTA kernel at high batch counts

Runs standalone on a single GPU or via torchrun.
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
ATTN_SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[rank={rank} gpu={gpu}] {msg}", file=sys.stderr, flush=True)


def call_fmha(q, kv_fp8, block_tables, seq_lens, bmm1_scale, bmm2_scale,
              device):
    """Call the TRT-LLM SM100 FMHA kernel via FlashInfer."""
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    bs = q.shape[0]
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    o = trtllm_batch_decode_with_kv_cache_mla(
        query=q.view(bs, 1, NUM_HEADS, HEAD_SIZE),
        kv_cache=kv_fp8.unsqueeze(1),
        workspace_buffer=workspace,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_lens.max().item(),
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
    )
    return o.view(bs, NUM_HEADS, KV_LORA_RANK)


def analyze_nan(out, label=""):
    """Detailed NaN analysis matching production log format."""
    nc = out.isnan().sum().item()
    if nc == 0:
        return 0, ""
    nan_mask = out.isnan().any(dim=-1).any(dim=-1)  # per-row
    nan_rows = nan_mask.sum().item()
    total_rows = out.shape[0]
    detail = (f"NaN={nc}/{out.numel()} ({nan_rows} rows of {total_rows}) "
              f"shape={list(out.shape)} dtype={out.dtype}")
    return nc, detail


# -----------------------------------------------------------------------
# Hypothesis A: FP8 value patterns
# -----------------------------------------------------------------------

def test_fp8_value_patterns(device):
    """Test kernel with specific FP8 value patterns that might trigger NaN.

    FP8 E4M3 has: denormals, zeros, near-max values (448), and limited precision.
    The kernel's dequant + softmax + accumulation might fail on certain patterns.
    """
    log("=== Hypothesis A: FP8 value patterns ===")
    fails = 0
    bs = 1024
    ctx = 256
    bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * bps + 2

    patterns = {
        "all_zeros": lambda n: torch.zeros(n, HEAD_SIZE, dtype=torch.float8_e4m3fn, device=device),
        "all_ones": lambda n: torch.ones(n, HEAD_SIZE, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn),
        "near_fp8_max": lambda n: (torch.ones(n, HEAD_SIZE, dtype=torch.bfloat16, device=device) * 400).to(torch.float8_e4m3fn),
        "at_fp8_max": lambda n: (torch.ones(n, HEAD_SIZE, dtype=torch.bfloat16, device=device) * FP8_MAX).to(torch.float8_e4m3fn),
        "fp8_denormals": lambda n: (torch.ones(n, HEAD_SIZE, dtype=torch.bfloat16, device=device) * 1e-4).to(torch.float8_e4m3fn),
        "mixed_sign_max": lambda n: (torch.where(
            torch.rand(n, HEAD_SIZE, device=device) < 0.5,
            torch.tensor(FP8_MAX, device=device),
            torch.tensor(-FP8_MAX, device=device)
        ).to(torch.bfloat16)).to(torch.float8_e4m3fn),
        "random_fp8": lambda n: torch.randn(n, HEAD_SIZE, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn),
        "sparse_max": lambda n: _sparse_max_pattern(n, HEAD_SIZE, device),
    }

    for pname, gen_fn in patterns.items():
        kv_fp8 = torch.zeros(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                              dtype=torch.float8_e4m3fn, device=device)
        block_tables = torch.zeros(bs, bps, dtype=torch.int32, device=device)

        next_block = 1
        for i in range(bs):
            for j in range(bps):
                if next_block < num_blocks:
                    block_tables[i, j] = next_block
                    fill = min(BLOCK_SIZE, ctx - j * BLOCK_SIZE)
                    kv_fp8[next_block, :fill] = gen_fn(fill)
                    next_block += 1

        seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)

        # Production-like scales
        q_scale = q.abs().max().item() / FP8_MAX
        k_scale = kv_fp8.to(torch.bfloat16).abs().max().item() / FP8_MAX
        bmm1 = max(q_scale, 1e-10) * max(k_scale, 1e-10) * ATTN_SCALE
        bmm2 = max(k_scale, 1e-10)

        try:
            out = call_fmha(q, kv_fp8, block_tables, seq_lens, bmm1, bmm2, device)
            nc, detail = analyze_nan(out)
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] {pname:20s} nan={nc:8d} "
                f"abs_max={am:.4f} bmm1={bmm1:.4e}")
            if detail:
                log(f"    {detail}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {pname}: {e}")

    return fails


# -----------------------------------------------------------------------
# Hypothesis B: Scale combinations
# -----------------------------------------------------------------------

def test_scale_combinations(device):
    """Test kernel with a grid of bmm1_scale × bmm2_scale values.

    The kernel multiplies attention scores by bmm1_scale before softmax,
    and output by bmm2_scale after. Extreme combinations could overflow
    intermediate FP32 or BF16 accumulators.
    """
    log("=== Hypothesis B: Scale combinations ===")
    fails = 0
    bs = 512
    ctx = 256
    bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * bps + 2

    kv_fp8, block_tables, seq_lens = _make_fp8_cache(bs, ctx, num_blocks, device)
    q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                     dtype=torch.bfloat16, device=device)

    bmm1_values = [1e-8, 1e-6, 1e-4, 1e-2, ATTN_SCALE, 0.1, 1.0, 10.0]
    bmm2_values = [1e-6, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]

    for bmm1 in bmm1_values:
        for bmm2 in bmm2_values:
            try:
                out = call_fmha(q, kv_fp8, block_tables, seq_lens,
                                bmm1, bmm2, device)
                nc, _ = analyze_nan(out)
                ok = nc == 0
                if not ok:
                    fails += 1
                    log(f"  [FAIL] bmm1={bmm1:.4e} bmm2={bmm2:.4e} nan={nc}")
            except Exception as e:
                fails += 1
                log(f"  [ERROR] bmm1={bmm1:.4e} bmm2={bmm2:.4e} {e}")

    log(f"  Grid: {len(bmm1_values)}x{len(bmm2_values)} = "
        f"{len(bmm1_values)*len(bmm2_values)} combos, {fails} failures")
    return fails


# -----------------------------------------------------------------------
# Hypothesis C: Batch size / seq_len edge cases
# -----------------------------------------------------------------------

def test_batch_seqlen_edge_cases(device):
    """Test kernel at batch/seqlen boundaries that might trigger edge cases.

    The 2-CTA persistent kernel partitions work across CTAs. Certain
    (batch_size, seq_len) combinations might leave one CTA with partial
    work, potentially reading uninitialized memory.
    """
    log("=== Hypothesis C: Batch/seqlen edge cases ===")
    fails = 0

    # Batch sizes around power-of-2 boundaries and CTA block boundaries
    cases = [
        # (bs, ctx) - production-relevant sizes
        (1, 1),
        (1, 32),
        (1, 33),    # misaligned to block_size
        (1, 4096),
        (31, 256),  # prime batch size
        (32, 256),
        (33, 256),
        (63, 256),
        (64, 256),
        (65, 256),
        (127, 256),
        (128, 256),
        (129, 256),
        (255, 256),
        (256, 256),
        (257, 256),
        (511, 256),
        (512, 256),
        (513, 256),
        (1023, 256),
        (1024, 256),  # production decode batch size
        (1025, 256),
        (2048, 256),
        (4096, 256),
        (512, 1),
        (512, 31),    # misaligned
        (512, 32),
        (512, 33),
        (512, 63),
        (512, 64),
        (512, 65),
        (512, 127),
        (512, 128),
        (512, 129),
        (512, 255),
        (512, 256),
        (512, 257),
        (512, 1023),
        (512, 1024),
        (512, 1025),
        (512, 4096),
    ]

    for bs, ctx in cases:
        bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * bps + 2

        try:
            kv_fp8, block_tables, seq_lens = _make_fp8_cache(bs, ctx, num_blocks, device)
            q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                             dtype=torch.bfloat16, device=device)
            bmm1, bmm2 = _prod_scales(q, kv_fp8)

            out = call_fmha(q, kv_fp8, block_tables, seq_lens, bmm1, bmm2, device)
            nc, detail = analyze_nan(out)
            ok = nc == 0
            if not ok:
                fails += 1
                log(f"  [FAIL] bs={bs:5d} ctx={ctx:5d} nan={nc} {detail}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] bs={bs} ctx={ctx} {e}")

    log(f"  {len(cases)} cases tested, {fails} failures")
    return fails


# -----------------------------------------------------------------------
# Hypothesis D: Non-contiguous block tables
# -----------------------------------------------------------------------

def test_noncontiguous_blocks(device):
    """Test kernel with randomized (non-contiguous) block table layouts.

    In production with NIXL KV transfer, blocks may be scattered.
    The kernel does indirect loads via block_tables — scattered blocks
    might stress the memory subsystem differently.
    """
    log("=== Hypothesis D: Non-contiguous block tables ===")
    fails = 0
    bs = 1024
    ctx = 256
    bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = bs * bps

    layouts = {
        "sequential": lambda: torch.arange(1, total_blocks + 1, device=device),
        "reversed": lambda: torch.arange(total_blocks, 0, -1, device=device),
        "random_perm": lambda: torch.randperm(total_blocks, device=device) + 1,
        "strided_2": lambda: _strided_blocks(total_blocks, 2, device),
        "strided_7": lambda: _strided_blocks(total_blocks, 7, device),
        "clustered": lambda: _clustered_blocks(total_blocks, 16, device),
    }

    for lname, gen_fn in layouts.items():
        num_blocks = total_blocks + 2
        kv_bf16 = torch.randn(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                                dtype=torch.bfloat16, device=device)
        kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

        block_ids = gen_fn()
        block_tables = torch.zeros(bs, bps, dtype=torch.int32, device=device)
        idx = 0
        for i in range(bs):
            for j in range(bps):
                block_tables[i, j] = block_ids[idx % len(block_ids)]
                idx += 1

        seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _prod_scales(q, kv_fp8)

        try:
            out = call_fmha(q, kv_fp8, block_tables, seq_lens, bmm1, bmm2, device)
            nc, detail = analyze_nan(out)
            ok = nc == 0
            if not ok:
                fails += 1
            am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
            log(f"  [{'PASS' if ok else 'FAIL'}] {lname:20s} nan={nc} abs_max={am:.4f}")
            if detail:
                log(f"    {detail}")
        except Exception as e:
            fails += 1
            log(f"  [ERROR] {lname}: {e}")

    return fails


# -----------------------------------------------------------------------
# Hypothesis E: Race condition at high batch counts
# -----------------------------------------------------------------------

def test_race_condition_stress(device):
    """Hammer the kernel at high batch counts to trigger potential races.

    The "Persistent2Cta" in the kernel name means 2 CTAs cooperate.
    At very high batch counts, CTA scheduling pressure increases,
    potentially exposing synchronization bugs.

    Run many iterations with fresh random data each time to maximize
    coverage of internal CTA scheduling patterns.
    """
    log("=== Hypothesis E: Race condition stress (1000 iters) ===")
    iters = 1000
    bs = 1024
    ctx = 256
    bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = bs * bps + 2

    nan_events = 0
    nan_details = []
    t0 = time.time()

    for it in range(iters):
        # Fresh random data every iteration
        kv_bf16 = torch.randn(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                                dtype=torch.bfloat16, device=device)
        kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

        block_tables = torch.zeros(bs, bps, dtype=torch.int32, device=device)
        # Random block permutation
        block_ids = torch.randperm(num_blocks - 1, device=device) + 1
        idx = 0
        for i in range(bs):
            for j in range(bps):
                block_tables[i, j] = block_ids[idx % len(block_ids)]
                idx += 1

        seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
        q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                         dtype=torch.bfloat16, device=device)
        bmm1, bmm2 = _prod_scales(q, kv_fp8)

        try:
            out = call_fmha(q, kv_fp8, block_tables, seq_lens, bmm1, bmm2, device)
            if out.isnan().any().item():
                nan_events += 1
                nc, detail = analyze_nan(out)
                if nan_events <= 5:
                    nan_details.append(f"iter={it} {detail}")
                    log(f"    NaN iter={it}: {detail}")
        except Exception as e:
            if it == 0:
                log(f"    KERNEL ERROR (aborting stress test): {e}")
                return 1
            nan_events += 1

        if (it + 1) % 200 == 0:
            elapsed = time.time() - t0
            log(f"    ... {it+1}/{iters}, {nan_events} NaN, {elapsed:.1f}s")

    status = "FAIL" if nan_events else "PASS"
    log(f"  [{status}] {nan_events}/{iters} NaN events")
    return 1 if nan_events > 0 else 0


# -----------------------------------------------------------------------
# Production-exact reproduction attempt
# -----------------------------------------------------------------------

def test_production_exact(device):
    """Reproduce the exact production scenario as closely as possible.

    Production: EP16 (4 nodes x 4 GPUs), DeepSeek-R1-0528-FP4-v2
    - Decode batch size ~1024 per GPU
    - Context length varies (post-prefill)
    - FP8 E4M3 KV cache with dynamic quantization
    - NIXL KV transfer from prefill nodes
    - Layer 1 attention (first non-embedding layer)
    - Queries come from: embed -> layer_norm -> qkv_proj -> q_norm -> rope -> W_UK BMM
    - KV cache from: embed -> layer_norm -> kv_proj -> kv_norm -> concat_and_cache_mla
    """
    log("=== Production-exact reproduction ===")
    fails = 0
    bs = 1024
    iters = 200

    for ctx in [256, 512, 1024, 2048]:
        bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        num_blocks = bs * bps + 2

        nan_events = 0
        for it in range(iters):
            # Simulate post-LN, post-kv_norm KV cache values (unit variance)
            kv_bf16 = torch.randn(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                                    dtype=torch.bfloat16, device=device)

            # Dynamic FP8 quantization (production path)
            kv_abs_max = kv_bf16.abs().max().item()
            kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

            # Sequential block tables (simplest case — still NaN in prod)
            block_tables = torch.zeros(bs, bps, dtype=torch.int32, device=device)
            next_block = 1
            for i in range(bs):
                for j in range(bps):
                    if next_block < num_blocks:
                        block_tables[i, j] = next_block
                        next_block += 1

            seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)

            # Simulate post-W_UK_BMM query (result of absorbed attention)
            # In production: q = q_nope @ W_UK_T, then concat with q_pe
            q = torch.randn(bs, NUM_HEADS, HEAD_SIZE,
                             dtype=torch.bfloat16, device=device)

            # Production scales
            q_abs_max = q.abs().max().item()
            q_scale = q_abs_max / FP8_MAX
            k_scale = kv_abs_max / FP8_MAX
            v_scale = kv_abs_max / FP8_MAX

            bmm1 = q_scale * k_scale * ATTN_SCALE
            bmm2 = v_scale

            try:
                out = call_fmha(q, kv_fp8, block_tables, seq_lens,
                                bmm1, bmm2, device)
                if out.isnan().any().item():
                    nan_events += 1
                    if nan_events <= 3:
                        nc, detail = analyze_nan(out)
                        log(f"    NaN ctx={ctx} iter={it}: {detail}")
            except Exception as e:
                if it == 0:
                    log(f"    KERNEL ERROR (aborting ctx={ctx}): {e}")
                    fails += 1
                    break
                nan_events += 1

            if (it + 1) % 50 == 0:
                log(f"    ctx={ctx} ... {it+1}/{iters}, {nan_events} NaN")

        status = "FAIL" if nan_events else "PASS"
        if not (nan_events == 0):
            fails += 1
        log(f"  [{status}] ctx={ctx}: {nan_events}/{iters} NaN")

    return fails


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_fp8_cache(bs, ctx, num_blocks, device):
    kv_bf16 = torch.randn(num_blocks, BLOCK_SIZE, HEAD_SIZE,
                            dtype=torch.bfloat16, device=device)
    kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)

    bps = (ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_tables = torch.zeros(bs, bps, dtype=torch.int32, device=device)
    next_block = 1
    for i in range(bs):
        for j in range(bps):
            if next_block < num_blocks:
                block_tables[i, j] = next_block
                next_block += 1

    seq_lens = torch.full((bs,), ctx, dtype=torch.int32, device=device)
    return kv_fp8, block_tables, seq_lens


def _prod_scales(q, kv_fp8):
    q_abs = q.abs().max().item()
    kv_abs = kv_fp8.to(torch.bfloat16).abs().max().item()
    q_s = max(q_abs / FP8_MAX, 1e-10)
    k_s = max(kv_abs / FP8_MAX, 1e-10)
    return q_s * k_s * ATTN_SCALE, max(kv_abs / FP8_MAX, 1e-10)


def _sparse_max_pattern(n, d, device):
    """Mostly zeros with a few near-max values."""
    data = torch.zeros(n, d, dtype=torch.bfloat16, device=device)
    mask = torch.rand(n, d, device=device) < 0.01
    data[mask] = FP8_MAX * (torch.randint(0, 2, (mask.sum().item(),),
                             device=device).float() * 2 - 1)
    return data.to(torch.float8_e4m3fn)


def _strided_blocks(total, stride, device):
    """Block IDs with a fixed stride (simulating fragmented allocation)."""
    ids = torch.arange(1, total * stride + 1, stride, device=device)
    return ids[:total]


def _clustered_blocks(total, cluster_size, device):
    """Blocks allocated in clusters with gaps between."""
    clusters = (total + cluster_size - 1) // cluster_size
    ids = []
    base = 1
    for c in range(clusters):
        for i in range(min(cluster_size, total - c * cluster_size)):
            ids.append(base + i)
        base += cluster_size * 3  # gap of 2x cluster between clusters
    return torch.tensor(ids[:total], device=device, dtype=torch.int32)


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    log(f"Device: {torch.cuda.get_device_name(device)}")
    props = torch.cuda.get_device_properties(device)
    log(f"SM: {props.major}.{props.minor}, {props.total_memory / 1e9:.0f}GB")

    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
        log("FlashInfer MLA kernel: available")
    except ImportError:
        log("SKIP: FlashInfer MLA not available (no trtllm_batch_decode_with_kv_cache_mla)")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0)

    total_fails = 0

    log("")
    log("=" * 72)
    log("TARGET: fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512...")
    log("=" * 72)

    log("")
    total_fails += test_fp8_value_patterns(device)

    log("")
    total_fails += test_scale_combinations(device)

    log("")
    total_fails += test_batch_seqlen_edge_cases(device)

    log("")
    total_fails += test_noncontiguous_blocks(device)

    log("")
    total_fails += test_race_condition_stress(device)

    log("")
    total_fails += test_production_exact(device)

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
