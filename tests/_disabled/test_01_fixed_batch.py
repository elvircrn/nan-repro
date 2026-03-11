#!/usr/bin/env python3
"""
Test 1: Fixed batch size CUDA graph replay.
Baseline — capture and replay at the same batch size every time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *


def main():
    h = DeepEPTestHarness()

    with h.vllm_ctx:
        for bs_label, bs in [("bs=1 (decode)", 1), (f"bs={MAX_TOKENS_PER_RANK} (full)", MAX_TOKENS_PER_RANK)]:
            h.log(f"\n=== Fixed {bs_label} ===")

            input_buf = torch.randn(bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
            topk_ids = torch.randint(0, NUM_EXPERTS, (bs, TOPK), device=h.device, dtype=torch.int64)
            topk_w = torch.ones(bs, TOPK, device=h.device, dtype=torch.float32)

            h.warmup(input_buf, topk_ids, topk_w)
            graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
            h.log("Graph captured.")

            nan_count = 0
            for i in range(NUM_REPLAYS):
                input_buf.copy_(torch.randn(bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
                topk_ids.copy_(torch.randint(0, NUM_EXPERTS, (bs, TOPK), device=h.device, dtype=torch.int64))

                graph.replay()
                torch.cuda.synchronize()

                if graph_out.isnan().any().item() or graph_out.isinf().any().item():
                    nan_count += 1
                    if nan_count == 1:
                        h.log_err(f"REPLAY {i} ({bs_label}): NaN={graph_out.isnan().sum().item()}")
                elif i % 50 == 0:
                    h.log(f"  Replay {i}/{NUM_REPLAYS}: OK")

            h.log_all(f"{bs_label}: {nan_count}/{NUM_REPLAYS} NaN")
            if nan_count > 0:
                h.teardown()
                sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
