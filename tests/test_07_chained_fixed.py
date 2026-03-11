#!/usr/bin/env python3
"""
Test 7: Chained MoE layers with fixed batch — tests 2, 4, 8, 16 layers.
Each layer's output feeds the next. All layers share one deep_ep.Buffer.
Uses NVFP4 + FlashInfer CuteDSL masked_gemm + fp8 dispatch (production path).
Tests whether buffer reuse / handle state across layers causes NaN.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.distributed
import deep_ep

from deepep_test_helpers import *

BATCH_SIZE = 64
REPLAYS = 200
LAYER_COUNTS = [2, 4, 8, 16]


def main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank % 4))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    init_workspace_manager(device)

    torch.distributed.init_process_group(backend="nccl", world_size=world_size)
    pg = torch.distributed.new_group(list(range(world_size)))

    num_local_experts = NUM_EXPERTS // world_size
    expert_map = torch.full((NUM_EXPERTS,), fill_value=-1, dtype=torch.int32)
    e_start = rank * num_local_experts
    expert_map[e_start:e_start + num_local_experts] = torch.arange(num_local_experts, dtype=torch.int32)
    expert_map = expert_map.to(device=device)

    moe_config = make_moe_config()
    total_fails = 0

    for n_layers in LAYER_COUNTS:
        vllm_ctx = set_current_vllm_config(VllmConfig())
        with vllm_ctx:
            if rank == 0:
                print(f"\n=== Chained fixed: {n_layers} layers, bs={BATCH_SIZE}, {REPLAYS} replays ===", flush=True)

            # Create shared buffer
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                BATCH_SIZE, HIDDEN_SIZE, world_size, NUM_EXPERTS)
            shared_buffer = deep_ep.Buffer(
                group=pg, num_rdma_bytes=num_rdma_bytes,
                low_latency_mode=True, num_qps_per_rank=num_local_experts)

            # Per-layer: a2a + FlashInfer CuteDSL experts + nvfp4 weights
            layers = []
            all_w1 = []
            all_w2 = []
            for _ in range(n_layers):
                mk, w1, w2 = make_moe_layer(
                    buffer=shared_buffer, world_size=world_size,
                    max_tokens_per_rank=BATCH_SIZE,
                    num_local_experts=num_local_experts,
                    moe_config=moe_config)
                layers.append(mk)
                all_w1.append(w1)
                all_w2.append(w2)

            input_buf = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) / 10
            topk_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=device, dtype=torch.int64)
            topk_w = torch.ones(BATCH_SIZE, TOPK, device=device, dtype=torch.float32)

            def chain_forward():
                x = input_buf
                for i, mk in enumerate(layers):
                    x = mk.apply(
                        hidden_states=x, w1=all_w1[i], w2=all_w2[i],
                        topk_weights=topk_w, topk_ids=topk_ids,
                        activation=MoEActivation.SILU, global_num_experts=NUM_EXPERTS,
                        expert_map=expert_map, apply_router_weight_on_input=False)
                return x

            # Warmup
            for _ in range(3):
                chain_forward()
            torch.cuda.synchronize()

            # Capture
            if rank == 0:
                print(f"Capturing CUDA graph with {n_layers} chained layers...", flush=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_out = chain_forward()
            torch.cuda.synchronize()

            # Replay
            nan_count = 0
            for i in range(REPLAYS):
                input_buf.copy_(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) / 10)
                topk_ids.copy_(torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=device, dtype=torch.int64))

                graph.replay()
                torch.cuda.synchronize()

                if graph_out.isnan().any().item() or graph_out.isinf().any().item():
                    nan_count += 1
                    if nan_count <= 5:
                        nc = graph_out.isnan().sum().item()
                        print(f"[RANK {rank}] REPLAY {i}: NaN={nc}/{graph_out.numel()}", file=sys.stderr, flush=True)
                elif i % 50 == 0 and rank == 0:
                    print(f"  Replay {i}/{REPLAYS}: OK", flush=True)

            print(f"[RANK {rank}] Chained {n_layers} layers: {nan_count}/{REPLAYS} NaN", flush=True)
            if nan_count > 0:
                total_fails += 1

            del graph, graph_out, layers, all_w1, all_w2, shared_buffer
            torch.cuda.empty_cache()

    torch.distributed.destroy_process_group()

    if total_fails > 0:
        sys.exit(1)
    if rank == 0:
        print("PASS", flush=True)


if __name__ == "__main__":
    main()
