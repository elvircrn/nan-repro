"""Stress-test vLLM MLA decode attention for NaN on Blackwell (GB200).

Uses the REAL vLLM MLA attention pipeline:
  - CUTLASS_MLA or FLASHINFER_MLA backend (SM100+)
  - concat_and_cache_mla KV cache writes
  - W_UK absorption BMM (q_nope x W_UK_T)
  - forward_mqa decode kernel
  - W_UV v_up_proj BMM (attn_out x W_UV)
  - FP8 KV cache (fp8, fp8_e4m3)

Production finding: NaN originates at layer 1 inside forward_mqa (decode
attention kernel) with ALL inputs clean:
  qkv_proj=0 q_norm=0 kv_norm=0 rope=0 kv_cache=0 mqa_q_pre=0
  fwd_mqa=65536 (NaN!)
Exactly 1 row (7168 elements) out of 1024 tokens affected.

Hypothesis: CUDA graph padding causes the decode kernel to process
extra tokens with garbage seq_lens/block_table entries, producing NaN.

This test exercises:
  1. Standard decode (baseline)
  2. Padded batches simulating CUDA graph padding
  3. Garbage in unused KV cache blocks
  4. Mixed prefill+decode batches
  5. FP8 KV cache variants

Runs via torchrun (4 nodes x 4 GPUs = 16 ranks) or single GPU.

Usage:
  torchrun --nnodes=4 --node-rank=$NODE_RANK --nproc-per-node=4 \
           --master-addr=$MASTER_ADDR --master-port=$PORT \
           test_mla_decode_nan.py

  # Single GPU:
  python test_mla_decode_nan.py
"""

import json
import os
import sys
import time

_vllm_src = os.environ.get("VLLM_SRC", "/opt/vllm-source")
if _vllm_src not in sys.path:
    sys.path.insert(0, _vllm_src)

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# DeepSeek-R1 MLA dimensions
# ---------------------------------------------------------------------------
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576

DEFAULT_BLOCK_SIZE = 32
MODEL = "deepseek-ai/DeepSeek-R1"

# Common CUDA graph padded sizes in vLLM
CUDA_GRAPH_PAD_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[rank={rank} gpu={gpu}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# vLLM config & metadata helpers
# ---------------------------------------------------------------------------

_base_config_cache = {}


def get_vllm_config(num_gpu_blocks, block_size, kv_cache_dtype="auto",
                    max_model_len=8192):
    import copy
    from tests.v1.attention.utils import create_vllm_config
    cache_key = (block_size, kv_cache_dtype)
    if cache_key not in _base_config_cache:
        cfg = create_vllm_config(
            model_name=MODEL,
            max_model_len=max_model_len,
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
        )
        cfg.cache_config.cache_dtype = kv_cache_dtype
        _base_config_cache[cache_key] = cfg
    vllm_config = copy.deepcopy(_base_config_cache[cache_key])
    vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
    vllm_config.model_config.max_model_len = max_model_len
    return vllm_config


def make_batch_metadata(seq_lens, query_lens, block_size, device,
                        num_actual_tokens=None):
    """Create attention metadata, optionally with padding.

    If num_actual_tokens is set, the metadata reports fewer actual tokens
    than the tensor sizes, simulating CUDA graph padding.
    """
    from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
    meta = create_common_attn_metadata(
        batch_spec=BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        block_size=block_size, device=device,
        max_block_idx=50000, arange_block_indices=True,
    )
    if num_actual_tokens is not None:
        meta.num_actual_tokens = num_actual_tokens
    return meta


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def make_kv_b_proj_weight(num_heads, device, dtype, scale=None):
    W_UK = torch.randn(KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM,
                       dtype=dtype, device=device)
    W_UV = torch.randn(KV_LORA_RANK, num_heads, V_HEAD_DIM,
                       dtype=dtype, device=device)
    if scale is None:
        scale = 1.0 / KV_LORA_RANK**0.5
    W_UK *= scale
    W_UV *= scale
    return torch.cat([W_UK, W_UV], dim=-1)


def load_real_kv_b_proj(layer_idx, num_heads, device, dtype):
    hf_cache = os.environ.get("HF_HOME", "/mnt/lustre/vllm-vlm-elvircrn")
    for snap_root in [
        f"{hf_cache}/hub/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots",
        f"{hf_cache}/models--nvidia--DeepSeek-R1-0528-FP4-v2/snapshots",
    ]:
        if not os.path.isdir(snap_root):
            continue
        snapshots = sorted(os.listdir(snap_root))
        if not snapshots:
            continue
        model_dir = os.path.join(snap_root, snapshots[-1])
        idx_file = os.path.join(model_dir, "model.safetensors.index.json")
        if not os.path.exists(idx_file):
            continue
        with open(idx_file) as f:
            wmap = json.load(f).get("weight_map", {})
        key = f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight"
        if key not in wmap:
            continue
        fpath = os.path.join(model_dir, wmap[key])
        if not os.path.exists(fpath):
            continue
        from safetensors.torch import load_file
        log(f"Loading real kv_b_proj layer={layer_idx} from {fpath}")
        w = load_file(fpath, device="cpu")[key].to(dtype=dtype, device=device)
        w = w.T.view(KV_LORA_RANK, num_heads, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        log(f"  shape={list(w.shape)} abs_max={w.abs().max().item():.4f} "
            f"abs_mean={w.abs().mean().item():.6f}")
        return w
    return None


# ---------------------------------------------------------------------------
# Core: run one MLA decode attention pass via real vLLM backend
# ---------------------------------------------------------------------------

class _MockParallelLinear(torch.nn.Module):
    """Mimics vLLM ColumnParallelLinear: forward returns (output, None)."""

    def __init__(self, weight, device, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype),
                                         requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight), None


def run_one(backend_enum, num_heads, seq_lens, query_lens,
            kv_b_proj_w, device, dtype, kv_cache_dtype="auto",
            block_size=DEFAULT_BLOCK_SIZE, pad_to=None,
            fill_unused_cache=None):
    """Run one attention pass.

    Args:
        pad_to: If set, pad tensors to this size but set num_actual_tokens
                to the real count, simulating CUDA graph padding.
        fill_unused_cache: If set, fill unused KV cache blocks with this
                          value (e.g. float('nan'), float('inf'), 1e4).
    """
    from tests.v1.attention.test_mla_backends import (
        create_and_prepopulate_kv_cache, run_attention_backend,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    block_align = 128 // block_size
    aligned_seq_lens = []
    for s in seq_lens:
        nblocks = (s + block_size - 1) // block_size
        nblocks = ((nblocks + block_align - 1) // block_align) * block_align
        aligned_seq_lens.append(nblocks * block_size)
    seq_lens = aligned_seq_lens

    bs = len(seq_lens)
    num_real_tokens = sum(query_lens)
    num_tokens = pad_to if pad_to and pad_to > num_real_tokens else num_real_tokens
    required_blocks = sum((s + block_size - 1) // block_size
                          for s in seq_lens) + 100
    required_blocks = ((required_blocks + block_align - 1) // block_align) * block_align

    vllm_config = get_vllm_config(
        required_blocks, block_size, kv_cache_dtype,
        max_model_len=max(seq_lens),
    )

    # Create metadata with actual token count (not padded)
    meta = make_batch_metadata(
        seq_lens, query_lens, block_size, device,
        num_actual_tokens=num_real_tokens if pad_to else None,
    )

    # Create input tensors at padded size
    q = torch.randn(num_tokens, num_heads, QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
                     dtype=dtype, device=device)
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device)

    # If padding, fill padded positions with garbage to simulate real conditions
    if pad_to and pad_to > num_real_tokens:
        q[num_real_tokens:] = torch.randn_like(q[num_real_tokens:]) * 100
        kv_c[num_real_tokens:] = torch.randn_like(kv_c[num_real_tokens:]) * 100
        k_pe[num_real_tokens:] = torch.randn_like(k_pe[num_real_tokens:]) * 100

    kv_c_contexts, k_pe_contexts = [], []
    for i in range(bs):
        ctx_len = seq_lens[i] - query_lens[i]
        kv_c_contexts.append(
            torch.randn(ctx_len, KV_LORA_RANK, dtype=dtype, device=device))
        k_pe_contexts.append(
            torch.randn(ctx_len, 1, QK_ROPE_HEAD_DIM, dtype=dtype, device=device))

    kv_cache = create_and_prepopulate_kv_cache(
        kv_c_contexts=kv_c_contexts, k_pe_contexts=k_pe_contexts,
        block_size=block_size, head_size=HEAD_SIZE, dtype=dtype,
        device=device, num_blocks=required_blocks,
        common_attn_metadata=meta, randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype if kv_cache_dtype != "auto" else None,
    )

    # Fill unused cache blocks with garbage if requested
    if fill_unused_cache is not None:
        used_blocks = sum((s + block_size - 1) // block_size for s in seq_lens)
        if used_blocks < required_blocks:
            unused = kv_cache[used_blocks:]
            if unused.dtype == torch.uint8:
                # FP8 stored as uint8: 0x7F = NaN in e4m3, 0x7E = max finite
                if fill_unused_cache != fill_unused_cache:  # NaN
                    unused.fill_(0x7F)
                elif fill_unused_cache == float('inf'):
                    unused.fill_(0x7E)  # max finite (FP8 e4m3 has no inf)
                else:
                    unused.fill_(0x7E)
            else:
                unused.fill_(fill_unused_cache)

    w_linear = kv_b_proj_w.reshape(
        KV_LORA_RANK, num_heads * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    ).T.contiguous()
    mock_kv_b_proj = _MockParallelLinear(w_linear, device, dtype)

    kv_spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1,
        head_size=HEAD_SIZE, dtype=dtype,
    )
    layer_names = ["model.layers.0.self_attn.attn"]

    torch.cuda.synchronize()
    t0 = time.time()
    try:
        out = run_attention_backend(
            backend=backend_enum, kv_cache_spec=kv_spec,
            layer_names=layer_names, vllm_config=vllm_config,
            device=device, common_attn_metadata=meta,
            query=q, kv_c=kv_c, k_pe=k_pe, kv_cache=kv_cache,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            mock_kv_b_proj=mock_kv_b_proj,
            kv_cache_dtype=kv_cache_dtype,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        # Check only real tokens (not padding) for NaN
        real_out = out[:num_real_tokens]
        nc = real_out.isnan().sum().item()
        ic = real_out.isinf().sum().item()
        safe = real_out[~real_out.isnan() & ~real_out.isinf()]
        am = safe.abs().max().item() if safe.numel() > 0 else float("nan")

        # Also check padded region separately
        pad_nan = 0
        if pad_to and pad_to > num_real_tokens:
            pad_out = out[num_real_tokens:]
            pad_nan = pad_out.isnan().sum().item()

        return dict(status="FAIL" if nc > 0 or ic > 0 else "PASS",
                    nan=nc, inf=ic, pad_nan=pad_nan,
                    abs_max=am, ms=elapsed * 1000, err=None)
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return dict(status="ERROR", nan=-1, inf=-1, pad_nan=-1,
                    abs_max=float("nan"), ms=-1, err=str(e))


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

def get_mla_backends():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    candidates = []
    for b in [AttentionBackendEnum.CUTLASS_MLA,
              AttentionBackendEnum.FLASHINFER_MLA]:
        try:
            b.get_class()
            candidates.append(b)
        except (ImportError, RuntimeError):
            pass
    return candidates


# ---------------------------------------------------------------------------
# Test suites
# ---------------------------------------------------------------------------

def _fmt(r):
    s = f"nan={r['nan']} inf={r['inf']}"
    if r.get('pad_nan', 0) > 0:
        s += f" pad_nan={r['pad_nan']}"
    s += f"  abs_max={r['abs_max']:.4f}"
    if r.get("ms") and r["ms"] > 0:
        s += f"  {r['ms']:.1f}ms"
    if r.get("err"):
        s += f"  ERR={r['err']}"
    return s


def test_baseline_decode(backend, num_heads, kv_b_w, device, dtype):
    """Test 1: Baseline decode without padding (should pass)."""
    log("--- Test 1: Baseline decode (no padding) ---")
    fails = 0
    for bs in [1, 32, 128, 256, 512, 1024]:
        r = run_one(backend, num_heads,
                    seq_lens=[257] * bs, query_lens=[1] * bs,
                    kv_b_proj_w=kv_b_w, device=device, dtype=dtype)
        if r["status"] != "PASS":
            fails += 1
        log(f"  [{r['status']}] bs={bs:5d}  {_fmt(r)}")
    return fails


def test_cuda_graph_padding(backend, num_heads, kv_b_w, device, dtype):
    """Test 2: Simulate CUDA graph padding.

    Create a batch with fewer actual tokens than the padded tensor size.
    This is the production scenario: CUDA graph captured at pad_size,
    replayed with num_actual_toks < pad_size.
    """
    log("--- Test 2: CUDA graph padding (actual < padded) ---")
    fails = 0
    ctx = 256
    for pad_size in CUDA_GRAPH_PAD_SIZES:
        # Try various actual batch sizes smaller than pad_size
        for actual_bs in [max(1, pad_size - 1), max(1, pad_size // 2),
                          max(1, pad_size - pad_size // 4)]:
            if actual_bs >= pad_size:
                continue
            r = run_one(backend, num_heads,
                        seq_lens=[ctx + 1] * actual_bs,
                        query_lens=[1] * actual_bs,
                        kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                        pad_to=pad_size)
            if r["status"] != "PASS":
                fails += 1
            log(f"  [{r['status']}] pad={pad_size:5d} actual={actual_bs:5d}  {_fmt(r)}")
    return fails


def test_garbage_cache_blocks(backend, num_heads, kv_b_w, device, dtype):
    """Test 3: Fill unused KV cache blocks with garbage values.

    In production, freed cache blocks retain old data. If the kernel
    reads past valid blocks (via bad block table or seq_len), it hits
    garbage.
    """
    log("--- Test 3: Garbage in unused KV cache blocks ---")
    fails = 0
    bs = 512
    ctx = 256
    for fill_val in [float('nan'), float('inf'), 1e4, -1e4]:
        r = run_one(backend, num_heads,
                    seq_lens=[ctx + 1] * bs, query_lens=[1] * bs,
                    kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                    fill_unused_cache=fill_val)
        if r["status"] != "PASS":
            fails += 1
        log(f"  [{r['status']}] fill={str(fill_val):6s}  {_fmt(r)}")
    return fails


def test_padding_with_garbage_cache(backend, num_heads, kv_b_w, device, dtype):
    """Test 4: CUDA graph padding + garbage in unused cache blocks.

    Combines the two most likely production conditions.
    """
    log("--- Test 4: Padding + garbage cache (production combo) ---")
    fails = 0
    ctx = 256
    for pad_size in [128, 256, 512, 1024]:
        for actual_bs in [max(1, pad_size - 1), max(1, pad_size // 2)]:
            if actual_bs >= pad_size:
                continue
            for fill_val in [float('nan'), float('inf')]:
                r = run_one(backend, num_heads,
                            seq_lens=[ctx + 1] * actual_bs,
                            query_lens=[1] * actual_bs,
                            kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                            pad_to=pad_size, fill_unused_cache=fill_val)
                if r["status"] != "PASS":
                    fails += 1
                log(f"  [{r['status']}] pad={pad_size} actual={actual_bs} "
                    f"fill={str(fill_val):6s}  {_fmt(r)}")
    return fails


def test_fp8_with_padding(backend, num_heads, kv_b_w, device, dtype):
    """Test 5: FP8 KV cache + CUDA graph padding."""
    log("--- Test 5: FP8 KV cache + padding ---")
    fails = 0
    ctx = 256
    for kv_dt in ["fp8", "fp8_e4m3"]:
        try:
            supported = kv_dt in backend.get_class().supported_kv_cache_dtypes
        except Exception:
            supported = False
        if not supported:
            log(f"  [SKIP] kv={kv_dt} (unsupported)")
            continue
        for pad_size in [256, 512, 1024]:
            actual_bs = pad_size - 1
            r = run_one(backend, num_heads,
                        seq_lens=[ctx + 1] * actual_bs,
                        query_lens=[1] * actual_bs,
                        kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                        kv_cache_dtype=kv_dt, pad_to=pad_size)
            if r["status"] != "PASS":
                fails += 1
            log(f"  [{r['status']}] kv={kv_dt:10s} pad={pad_size} "
                f"actual={actual_bs}  {_fmt(r)}")
    return fails


def test_mixed_prefill_decode_padded(backend, num_heads, kv_b_w, device, dtype):
    """Test 6: Mixed prefill + decode with CUDA graph padding.

    Production uses chunked prefill where some tokens are prefill
    (query_len > 1) and some are decode (query_len = 1).
    """
    log("--- Test 6: Mixed prefill+decode with padding ---")
    fails = 0
    ctx = 256
    # Mix: some decode (query_len=1), some prefill (query_len>1)
    configs = [
        # (num_decode, num_prefill, prefill_qlen)
        (100, 5, 10),
        (200, 10, 20),
        (500, 5, 50),
    ]
    for n_dec, n_pre, pre_qlen in configs:
        seq_lens = [ctx + 1] * n_dec + [ctx + pre_qlen] * n_pre
        query_lens = [1] * n_dec + [pre_qlen] * n_pre
        num_real = sum(query_lens)
        # Pad to next CUDA graph size
        pad_to = None
        for ps in CUDA_GRAPH_PAD_SIZES:
            if ps > num_real:
                pad_to = ps
                break
        if pad_to is None:
            pad_to = num_real + 64

        r = run_one(backend, num_heads,
                    seq_lens=seq_lens, query_lens=query_lens,
                    kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                    pad_to=pad_to)
        if r["status"] != "PASS":
            fails += 1
        log(f"  [{r['status']}] dec={n_dec} pre={n_pre}x{pre_qlen} "
            f"real={num_real} pad={pad_to}  {_fmt(r)}")
    return fails


def test_stress_with_padding(backend, num_heads, kv_b_w, device, dtype):
    """Test 7: Stress test with padding — many iterations."""
    log("--- Test 7: Stress test with padding ---")
    total_nan_events = 0
    iters = 200
    ctx = 256
    for pad_size in [512, 1024]:
        actual_bs = pad_size - 1
        nan_events = 0
        log(f"  Stress pad={pad_size} actual={actual_bs}, {iters} iters")
        for it in range(iters):
            r = run_one(backend, num_heads,
                        seq_lens=[ctx + 1] * actual_bs,
                        query_lens=[1] * actual_bs,
                        kv_b_proj_w=kv_b_w, device=device, dtype=dtype,
                        pad_to=pad_size)
            if r["status"] != "PASS":
                nan_events += 1
                if nan_events <= 5:
                    log(f"    NaN iter={it} {_fmt(r)}")
            if (it + 1) % 50 == 0:
                log(f"    ... {it+1}/{iters}, {nan_events} NaN")
        status = "FAIL" if nan_events else "PASS"
        log(f"  [{status}] pad={pad_size}: {nan_events}/{iters} NaN events")
        total_nan_events += nan_events
    return total_nan_events


def test_real_weights_with_padding(backend, num_heads, device, dtype):
    """Test 8: Real NVFP4 weights + padding."""
    log("--- Test 8: Real weights + padding ---")
    fails = 0
    ctx = 256
    for layer_idx in [0, 1]:
        w = load_real_kv_b_proj(layer_idx, num_heads, device, dtype)
        if w is None:
            log(f"  [SKIP] layer={layer_idx} (weights not on Lustre)")
            continue
        for pad_size in [256, 512, 1024]:
            actual_bs = pad_size - 1
            r = run_one(backend, num_heads,
                        seq_lens=[ctx + 1] * actual_bs,
                        query_lens=[1] * actual_bs,
                        kv_b_proj_w=w, device=device, dtype=dtype,
                        pad_to=pad_size)
            if r["status"] != "PASS":
                fails += 1
            log(f"  [{r['status']}] layer={layer_idx} pad={pad_size} "
                f"actual={actual_bs}  {_fmt(r)}")
    return fails


def test_fp8_real_weights_padded(backend, num_heads, device, dtype):
    """Test 9: FP8 KV cache + real weights + padding — closest to production."""
    log("--- Test 9: FP8 + real weights + padding (production config) ---")
    fails = 0
    ctx = 256
    for layer_idx in [0, 1]:
        w = load_real_kv_b_proj(layer_idx, num_heads, device, dtype)
        if w is None:
            log(f"  [SKIP] layer={layer_idx} (weights not on Lustre)")
            continue
        for kv_dt in ["fp8", "fp8_e4m3"]:
            try:
                supported = kv_dt in backend.get_class().supported_kv_cache_dtypes
            except Exception:
                supported = False
            if not supported:
                continue
            for pad_size in [256, 512, 1024]:
                actual_bs = pad_size - 1
                r = run_one(backend, num_heads,
                            seq_lens=[ctx + 1] * actual_bs,
                            query_lens=[1] * actual_bs,
                            kv_b_proj_w=w, device=device, dtype=dtype,
                            kv_cache_dtype=kv_dt, pad_to=pad_size,
                            fill_unused_cache=float('nan'))
                if r["status"] != "PASS":
                    fails += 1
                log(f"  [{r['status']}] layer={layer_idx} kv={kv_dt} "
                    f"pad={pad_size} actual={actual_bs}  {_fmt(r)}")
    return fails


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    sm = torch.cuda.get_device_properties(device)
    log(f"Device: {torch.cuda.get_device_name(device)} "
        f"SM{sm.major}.{sm.minor} "
        f"{sm.total_memory / 1e9:.0f}GB")

    backends = get_mla_backends()
    if not backends:
        log("ERROR: No MLA backends available")
        sys.exit(1)
    backend = backends[0]
    log(f"Primary backend: {backend.name}")
    log(f"All backends: {[b.name for b in backends]}")

    num_heads = 128
    try:
        vllm_config = get_vllm_config(1000, DEFAULT_BLOCK_SIZE)
        num_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        log(f"Model num_heads from config: {num_heads}")
    except Exception as e:
        log(f"Could not load model config, using num_heads={num_heads}: {e}")

    kv_b_w = make_kv_b_proj_weight(num_heads, device, dtype)

    total_fails = 0

    # Test 1: Baseline (no padding)
    log("\n" + "=" * 72)
    total_fails += test_baseline_decode(backend, num_heads, kv_b_w, device, dtype)

    # Test 2: CUDA graph padding
    log("\n" + "=" * 72)
    total_fails += test_cuda_graph_padding(backend, num_heads, kv_b_w, device, dtype)

    # Test 3: Garbage cache blocks
    log("\n" + "=" * 72)
    total_fails += test_garbage_cache_blocks(backend, num_heads, kv_b_w, device, dtype)

    # Test 4: Padding + garbage cache
    log("\n" + "=" * 72)
    total_fails += test_padding_with_garbage_cache(backend, num_heads, kv_b_w,
                                                    device, dtype)

    # Test 5: FP8 + padding
    log("\n" + "=" * 72)
    total_fails += test_fp8_with_padding(backend, num_heads, kv_b_w, device, dtype)

    # Test 6: Mixed prefill+decode with padding
    log("\n" + "=" * 72)
    total_fails += test_mixed_prefill_decode_padded(backend, num_heads, kv_b_w,
                                                     device, dtype)

    # Test 7: Stress with padding
    log("\n" + "=" * 72)
    total_fails += test_stress_with_padding(backend, num_heads, kv_b_w,
                                             device, dtype)

    # Test 8: Real weights + padding
    log("\n" + "=" * 72)
    total_fails += test_real_weights_with_padding(backend, num_heads, device, dtype)

    # Test 9: Full production config
    log("\n" + "=" * 72)
    total_fails += test_fp8_real_weights_padded(backend, num_heads, device, dtype)

    log("\n" + "=" * 72)
    log(f"TOTAL FAILURES: {total_fails}")
    log("RESULT: " + ("FAIL" if total_fails else "PASS"))
    log("=" * 72)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    main()
