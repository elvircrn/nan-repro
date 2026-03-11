#!/usr/bin/env python3
"""
Test 3: Graph captured at MAX, replayed with random effective_bs each time.
Zero-pads unused rows. Checks only real output rows for NaN.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *


def main():
    h = DeepEPTestHarness()
    capture_bs = MAX_TOKENS_PER_RANK

    with h.vllm_ctx:
        h.log(f"\n=== Varying effective_bs (1..{capture_bs}) ===")

        input_buf = torch.randn(capture_bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (capture_bs, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(capture_bs, TOPK, device=h.device, dtype=torch.float32)

        h.warmup(input_buf, topk_ids, topk_w)
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured.")

        nan_count = 0
        for i in range(NUM_REPLAYS):
            effective_bs = torch.randint(1, capture_bs + 1, (1,)).item()

            input_buf.zero_()
            topk_ids.zero_()
            input_buf[:effective_bs].copy_(
                torch.randn(effective_bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
            topk_ids[:effective_bs].copy_(
                torch.randint(0, NUM_EXPERTS, (effective_bs, TOPK), device=h.device, dtype=torch.int64))

            graph.replay()
            torch.cuda.synchronize()

            real_out = graph_out[:effective_bs]
            if real_out.isnan().any().item() or real_out.isinf().any().item():
                nan_count += 1
                if nan_count <= 3:
                    h.log_err(f"REPLAY {i}: effective_bs={effective_bs} "
                              f"NaN={real_out.isnan().sum().item()}/{real_out.numel()}")
            elif i % 50 == 0:
                h.log(f"  Replay {i}/{NUM_REPLAYS}: effective_bs={effective_bs} OK")

        h.log_all(f"Varying bs: {nan_count}/{NUM_REPLAYS} NaN")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
