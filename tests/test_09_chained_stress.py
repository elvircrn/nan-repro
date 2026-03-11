#!/usr/bin/env python3
"""
Test 9: Chained MoE stress — 16 layers, bs=256, 500 replays.
High concurrency + many layers to maximize buffer contention.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *

BATCH_SIZE = 256
NUM_LAYERS = 16
REPLAYS = 500


def main():
    h = ChainedMoEHarness(num_layers=NUM_LAYERS, max_tokens_per_rank=BATCH_SIZE)

    with h.vllm_ctx:
        h.log(f"\n=== Chained stress: {NUM_LAYERS} layers, bs={BATCH_SIZE}, {REPLAYS} replays ===")

        input_buf = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(BATCH_SIZE, TOPK, device=h.device, dtype=torch.float32)

        h.log("Warming up...")
        h.warmup(input_buf, topk_ids, topk_w)

        h.log(f"Capturing CUDA graph with {NUM_LAYERS} chained layers at bs={BATCH_SIZE}...")
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured. Starting stress test...")

        nan_count = 0
        for i in range(REPLAYS):
            input_buf.copy_(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
            topk_ids.copy_(torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=h.device, dtype=torch.int64))

            graph.replay()
            torch.cuda.synchronize()

            if graph_out.isnan().any().item() or graph_out.isinf().any().item():
                nan_count += 1
                if nan_count <= 5:
                    nc = graph_out.isnan().sum().item()
                    h.log_err(f"REPLAY {i}: NaN={nc}/{graph_out.numel()}")
            elif i % 100 == 0:
                h.log(f"  Replay {i}/{REPLAYS}: OK")

        h.log_all(f"Chained stress {NUM_LAYERS}L bs={BATCH_SIZE}: {nan_count}/{REPLAYS} NaN")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
