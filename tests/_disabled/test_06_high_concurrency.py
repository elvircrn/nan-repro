#!/usr/bin/env python3
"""
Test 6: High concurrency stress test — bs=1024, 1000 replays.
Simulates sustained heavy decode load to trigger buffer/NVSHMEM corruption
that only manifests under pressure.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *

STRESS_BS = 1024
STRESS_REPLAYS = 1000
STRESS_MAX_TOKENS = STRESS_BS  # must match capture size


def main():
    h = DeepEPTestHarness(max_tokens_per_rank=STRESS_MAX_TOKENS)

    with h.vllm_ctx:
        h.log(f"\n=== High concurrency: bs={STRESS_BS}, {STRESS_REPLAYS} replays ===")

        input_buf = torch.randn(STRESS_BS, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (STRESS_BS, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(STRESS_BS, TOPK, device=h.device, dtype=torch.float32)

        h.log("Warming up...")
        h.warmup(input_buf, topk_ids, topk_w)

        h.log(f"Capturing CUDA graph at bs={STRESS_BS}...")
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured. Starting stress test...")

        nan_count = 0
        for i in range(STRESS_REPLAYS):
            input_buf.copy_(torch.randn(STRESS_BS, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
            topk_ids.copy_(torch.randint(0, NUM_EXPERTS, (STRESS_BS, TOPK), device=h.device, dtype=torch.int64))

            graph.replay()
            torch.cuda.synchronize()

            if graph_out.isnan().any().item() or graph_out.isinf().any().item():
                nan_count += 1
                if nan_count <= 5:
                    nc = graph_out.isnan().sum().item()
                    h.log_err(f"REPLAY {i}: NaN={nc}/{graph_out.numel()}")
            elif i % 200 == 0:
                h.log(f"  Replay {i}/{STRESS_REPLAYS}: OK")

        h.log_all(f"High concurrency bs={STRESS_BS}: {nan_count}/{STRESS_REPLAYS} NaN")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
