#!/usr/bin/env python3
"""
Test 10: Full 61-layer chained MoE — matches DeepSeek-R1 (layers 3-63).
All 61 layers share one deep_ep.Buffer, each with its own
DeepEPLLPrepareAndFinalize (and thus its own handles[] state).
NVFP4 + FlashInfer CuteDSL masked_gemm + NVFP4 dispatch.

bs=1024, runs indefinitely until NaN is found.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import torch
from deepep_test_helpers import *

BATCH_SIZE = 1024
NUM_LAYERS = 61  # DeepSeek-R1: layers 3-63 are MoE


def main():
    h = ChainedMoEHarness(num_layers=NUM_LAYERS, max_tokens_per_rank=BATCH_SIZE)

    with h.vllm_ctx:
        h.log(f"\n=== Full 61-layer chain: bs={BATCH_SIZE}, running until NaN ===")
        h.log(f"Created {NUM_LAYERS} layers, each with own a2a, sharing 1 buffer")

        input_buf = torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(BATCH_SIZE, TOPK, device=h.device, dtype=torch.float32)

        h.log("Warming up...")
        h.warmup(input_buf, topk_ids, topk_w)

        h.log(f"Capturing CUDA graph with {NUM_LAYERS} chained layers at bs={BATCH_SIZE}...")
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured. Replaying until NaN...")

        nan_count = 0
        i = 0
        t0 = time.time()
        while True:
            input_buf.copy_(torch.randn(BATCH_SIZE, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
            topk_ids.copy_(torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOPK), device=h.device, dtype=torch.int64))

            graph.replay()
            torch.cuda.synchronize()

            if graph_out.isnan().any().item() or graph_out.isinf().any().item():
                nan_count += 1
                nc = graph_out.isnan().sum().item()
                ic = graph_out.isinf().sum().item()
                h.log_all(f"REPLAY {i}: NaN={nc} Inf={ic} / {graph_out.numel()}")
                if nan_count >= 10:
                    h.log_all(f"STOPPING after {nan_count} NaN in {i+1} replays")
                    break
            elif i % 100 == 0:
                elapsed = time.time() - t0
                h.log(f"  Replay {i}: OK ({elapsed:.1f}s elapsed, {nan_count} NaN so far)")

            i += 1

        elapsed = time.time() - t0
        h.log_all(f"Full 61-layer chain bs={BATCH_SIZE}: {nan_count} NaN in {i+1} replays ({elapsed:.1f}s)")

    h.teardown()
    if nan_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
