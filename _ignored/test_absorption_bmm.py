"""Stress-test the MLA absorption BMMs (W_UK and W_UV) for NaN.

In the decode path, MLAAttention.forward_impl does:
  1. mqa_ql_nope = torch.bmm(q_nope, W_UK_T)  # (N,B,P) x (N,P,L) -> (N,B,L)
  2. attn_out = forward_mqa(...)               # attention kernel
  3. output = torch.bmm(attn_out, W_UV)        # (N,B,L) x (N,L,V) -> (N,B,V)

W_UK_T and W_UV come from dequantized kv_b_proj (NVFP4 -> bf16/fp16).
This test exercises these BMMs in isolation at production dimensions
and batch sizes, with both random and real (from checkpoint) weights.

Key concern: dequantized NVFP4 weights can have unusual value
distributions (limited to FP4 grid values × block scales). Combined
with post-RoPE query vectors, the BMM accumulation might overflow
or produce NaN.

Runs via torchrun (4 nodes x 4 GPUs = 16 ranks).
"""

import json
import os
import sys

import torch
import torch.distributed as dist

NUM_HEADS = 128
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128


def log(msg: str):
    rank = dist.get_rank() if dist.is_initialized() else 0
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[rank={rank} gpu={gpu}] {msg}", file=sys.stderr, flush=True)


def load_real_kv_b_proj(layer_idx, device, dtype):
    """Load real kv_b_proj weight and split into W_UK, W_UV."""
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
        w = load_file(fpath, device="cpu")[key].to(dtype=dtype, device=device)
        # (out, in) = (N*(P+V), L) -> (L, N, P+V)
        w = w.T.view(KV_LORA_RANK, NUM_HEADS, QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        W_UK, W_UV = w.split([QK_NOPE_HEAD_DIM, V_HEAD_DIM], dim=-1)
        return W_UK, W_UV
    return None, None


def make_random_weights(device, dtype, scale=None):
    W_UK = torch.randn(KV_LORA_RANK, NUM_HEADS, QK_NOPE_HEAD_DIM,
                       dtype=dtype, device=device)
    W_UV = torch.randn(KV_LORA_RANK, NUM_HEADS, V_HEAD_DIM,
                       dtype=dtype, device=device)
    if scale is None:
        scale = 1.0 / KV_LORA_RANK**0.5
    return W_UK * scale, W_UV * scale


def run_w_uk_bmm(q_nope, W_UK_T):
    """Replicate the W_UK absorption BMM from forward_impl."""
    # q_nope: (B, N, P) -> transpose to (N, B, P)
    q_t = q_nope.transpose(0, 1)
    # W_UK_T: (N, P, L)
    # Result: (N, B, L) -> transpose to (B, N, L)
    return torch.bmm(q_t, W_UK_T).transpose(0, 1)


def run_w_uv_bmm(attn_out, W_UV):
    """Replicate the W_UV v_up_proj BMM from forward_impl."""
    # attn_out: (B, N, L) -> transpose to (N, B, L)
    x = attn_out.transpose(0, 1)
    # W_UV: (N, L, V)
    # Result: (N, B, V) -> transpose to (B, N, V)
    return torch.bmm(x, W_UV).transpose(0, 1)


def test_w_uk_bmm_batch_sweep(W_UK, device, dtype):
    """Test W_UK BMM at increasing batch sizes."""
    log("--- W_UK BMM batch sweep ---")
    # W_UK: (L, N, P) -> W_UK_T: (N, P, L)
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()
    fails = 0

    for bs in [1, 64, 256, 512, 1024, 2048, 4096, 8192]:
        q_nope = torch.randn(bs, NUM_HEADS, QK_NOPE_HEAD_DIM,
                             dtype=dtype, device=device)
        out = run_w_uk_bmm(q_nope, W_UK_T)
        nc = out.isnan().sum().item()
        ic = out.isinf().sum().item()
        am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
        ok = nc == 0 and ic == 0
        if not ok:
            fails += 1
        log(f"  [{'PASS' if ok else 'FAIL'}] bs={bs:5d}  "
            f"nan={nc} inf={ic} abs_max={am:.4f}")

    return fails


def test_w_uv_bmm_batch_sweep(W_UV, device, dtype):
    """Test W_UV BMM at increasing batch sizes."""
    log("--- W_UV BMM batch sweep ---")
    # W_UV: (L, N, V) -> (N, L, V)
    W_UV_t = W_UV.transpose(0, 1).contiguous()
    fails = 0

    for bs in [1, 64, 256, 512, 1024, 2048, 4096, 8192]:
        # attn_out shape from forward_mqa: (B, N, L)
        attn_out = torch.randn(bs, NUM_HEADS, KV_LORA_RANK,
                               dtype=dtype, device=device)
        out = run_w_uv_bmm(attn_out, W_UV_t)
        nc = out.isnan().sum().item()
        ic = out.isinf().sum().item()
        am = out[~out.isnan()].abs().max().item() if nc < out.numel() else float("nan")
        ok = nc == 0 and ic == 0
        if not ok:
            fails += 1
        log(f"  [{'PASS' if ok else 'FAIL'}] bs={bs:5d}  "
            f"nan={nc} inf={ic} abs_max={am:.4f}")

    return fails


def test_bmm_input_distributions(W_UK, W_UV, device, dtype):
    """Test BMMs with various input distributions mimicking production."""
    log("--- BMM input distribution sweep ---")
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()
    W_UV_t = W_UV.transpose(0, 1).contiguous()
    fails = 0
    bs = 1024

    distributions = {
        "normal_std1": lambda s: torch.randn(*s, dtype=dtype, device=device),
        "normal_std0.01": lambda s: torch.randn(*s, dtype=dtype, device=device) * 0.01,
        "normal_std100": lambda s: torch.randn(*s, dtype=dtype, device=device) * 100,
        "uniform_-1_1": lambda s: torch.rand(*s, dtype=dtype, device=device) * 2 - 1,
        "post_rmsnorm": lambda s: _post_rmsnorm(*s, dtype=dtype, device=device),
        "sparse_outlier": lambda s: _sparse(*s, dtype=dtype, device=device),
        "all_same_sign": lambda s: torch.randn(*s, dtype=dtype, device=device).abs(),
        "bimodal": lambda s: _bimodal(*s, dtype=dtype, device=device),
    }

    for name, gen in distributions.items():
        # W_UK BMM
        q_nope = gen((bs, NUM_HEADS, QK_NOPE_HEAD_DIM))
        uk_out = run_w_uk_bmm(q_nope, W_UK_T)
        uk_nan = uk_out.isnan().sum().item()

        # W_UV BMM
        attn_out = gen((bs, NUM_HEADS, KV_LORA_RANK))
        uv_out = run_w_uv_bmm(attn_out, W_UV_t)
        uv_nan = uv_out.isnan().sum().item()

        ok = uk_nan == 0 and uv_nan == 0
        if not ok:
            fails += 1
        log(f"  [{'PASS' if ok else 'FAIL'}] {name:20s}  "
            f"W_UK_nan={uk_nan} W_UV_nan={uv_nan}  "
            f"W_UK_abs_max={uk_out.abs().max().item():.4f} "
            f"W_UV_abs_max={uv_out.abs().max().item():.4f}")

    return fails


def test_bmm_stress(W_UK, W_UV, device, dtype):
    """Repeated BMMs to catch intermittent NaN."""
    log("--- BMM stress test ---")
    W_UK_T = W_UK.permute(1, 2, 0).contiguous()
    W_UV_t = W_UV.transpose(0, 1).contiguous()
    iters = 500
    bs = 4096

    uk_nans = 0
    uv_nans = 0
    for it in range(iters):
        q_nope = torch.randn(bs, NUM_HEADS, QK_NOPE_HEAD_DIM,
                             dtype=dtype, device=device)
        uk_out = run_w_uk_bmm(q_nope, W_UK_T)
        if uk_out.isnan().any().item():
            uk_nans += 1

        attn_out = torch.randn(bs, NUM_HEADS, KV_LORA_RANK,
                               dtype=dtype, device=device)
        uv_out = run_w_uv_bmm(attn_out, W_UV_t)
        if uv_out.isnan().any().item():
            uv_nans += 1

        if (it + 1) % 100 == 0:
            log(f"    ... {it+1}/{iters}, W_UK_nans={uk_nans} W_UV_nans={uv_nans}")

    status = "FAIL" if (uk_nans + uv_nans) > 0 else "PASS"
    log(f"  [{status}] bs={bs} {iters} iters: "
        f"W_UK_nans={uk_nans} W_UV_nans={uv_nans}")
    return 1 if (uk_nans + uv_nans) > 0 else 0


def _post_rmsnorm(*shape, dtype, device):
    x = torch.randn(*shape, dtype=torch.float32, device=device)
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
    return (x / rms).to(dtype)


def _sparse(*shape, dtype, device):
    x = torch.zeros(*shape, dtype=dtype, device=device)
    mask = torch.rand(*shape, device=device) < 0.01
    x[mask] = torch.randn(mask.sum().item(), dtype=dtype, device=device) * 50
    return x


def _bimodal(*shape, dtype, device):
    x = torch.randn(*shape, dtype=dtype, device=device)
    mask = torch.rand(*shape, device=device) < 0.5
    x[mask] *= 10
    x[~mask] *= 0.1
    return x


def main():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        local_rank = 0

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16

    log(f"Device: {torch.cuda.get_device_name(device)}")

    total_fails = 0

    # Test with random weights first
    log("")
    log("=" * 72)
    log("Random weights (scaled 1/sqrt(L))")
    log("=" * 72)
    W_UK, W_UV = make_random_weights(device, dtype)
    total_fails += test_w_uk_bmm_batch_sweep(W_UK, device, dtype)
    total_fails += test_w_uv_bmm_batch_sweep(W_UV, device, dtype)
    total_fails += test_bmm_input_distributions(W_UK, W_UV, device, dtype)
    total_fails += test_bmm_stress(W_UK, W_UV, device, dtype)

    # Test with real weights from layers 0, 1, 2
    for layer_idx in [0, 1, 2]:
        log("")
        log("=" * 72)
        log(f"Real NVFP4 weights layer={layer_idx}")
        log("=" * 72)
        W_UK_real, W_UV_real = load_real_kv_b_proj(layer_idx, device, dtype)
        if W_UK_real is None:
            log("  [SKIP] weights not on Lustre")
            continue
        log(f"  W_UK abs_max={W_UK_real.abs().max().item():.4f} "
            f"abs_mean={W_UK_real.abs().mean().item():.6f}")
        log(f"  W_UV abs_max={W_UV_real.abs().max().item():.4f} "
            f"abs_mean={W_UV_real.abs().mean().item():.6f}")
        total_fails += test_w_uk_bmm_batch_sweep(W_UK_real, device, dtype)
        total_fails += test_w_uv_bmm_batch_sweep(W_UV_real, device, dtype)
        total_fails += test_bmm_input_distributions(W_UK_real, W_UV_real,
                                                     device, dtype)
        total_fails += test_bmm_stress(W_UK_real, W_UV_real, device, dtype)

    # Test with extreme weight scales (simulate bad NVFP4 dequant)
    log("")
    log("=" * 72)
    log("Extreme weight scale sweep")
    log("=" * 72)
    for exp in [-4, -2, 0, 2, 4]:
        scale = 10.0 ** exp
        W_UK_s, W_UV_s = make_random_weights(device, dtype, scale=scale)
        log(f"  scale=1e{exp:+d}")
        total_fails += test_w_uk_bmm_batch_sweep(W_UK_s, device, dtype)
        total_fails += test_w_uv_bmm_batch_sweep(W_UV_s, device, dtype)

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
