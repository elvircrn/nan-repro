#!/usr/bin/env python3
"""
Test 8: Chained MoE layers with varying effective batch size.
Graph captured at bs=64 with 8 layers. Each replay uses random effective_bs,
zero-padding the rest. This is the realistic decode scenario: batch size
varies per step but graph is fixed-size.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *

MAX_BS = 64
NUM_LAYERS = 8
REPLAYS = 200


def main():
    h = ChainedMoEHarness(num_layers=NUM_LAYERS, max_tokens_per_rank=MAX_BS)

    with h.vllm_ctx:
        h.log(f"\n=== Chained varying bs: {NUM_LAYERS} layers, max_bs={MAX_BS}, {REPLAYS} replays ===")

        input_buf = torch.randn(MAX_BS, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (MAX_BS, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(MAX_BS, TOPK, device=h.device, dtype=torch.float32)

        h.log("Warming up...")
        h.warmup(input_buf, topk_ids, topk_w)

        h.log(f"Capturing CUDA graph at bs={MAX_BS}...")
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured. Starting varying-bs replays...")

        nan_count = 0
        for i in range(REPLAYS):
            effective_bs = torch.randint(1, MAX_BS + 1, (1,)).item()

            input_buf.zero_()
            topk_ids.zero_()
            topk_w.zero_()

            input_buf[:effective_bs].copy_(
                torch.randn(effective_bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
            )
            topk_ids[:effective_bs].copy_(
                torch.randint(0, NUM_EXPERTS, (effective_bs, TOPK), device=h.device, dtype=torch.int64)
            )
            topk_w[:effective_bs] = 1.0

            graph.replay()
            torch.cuda.synchronize()

            active_out = graph_out[:effective_bs]
            if active_out.isnan().any().item() or active_out.isinf().any().item():
                nan_count += 1
                if nan_count <= 5:
                    nc = active_out.isnan().sum().item()
                    h.log_err(f"REPLAY {i} (eff_bs={effective_bs}): NaN={nc}/{active_out.numel()}")
            elif i % 50 == 0:
                h.log(f"  Replay {i}/{REPLAYS} (eff_bs={effective_bs}): OK")

        h.log_all(f"Chained {NUM_LAYERS} layers varying bs: {nan_count}/{REPLAYS} NaN")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
