#!/usr/bin/env python3
"""
Test 2: Graph captured at MAX batch size, replayed with effective_bs=1.
Only first row has real data, rest zero-padded.
Checks if padding rows leak NaN into real output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *


def main():
    h = DeepEPTestHarness()
    capture_bs = MAX_TOKENS_PER_RANK

    with h.vllm_ctx:
        h.log(f"\n=== Padded bs=1 (captured at {capture_bs}) ===")

        input_buf = torch.randn(capture_bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (capture_bs, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(capture_bs, TOPK, device=h.device, dtype=torch.float32)

        h.warmup(input_buf, topk_ids, topk_w)
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured.")

        nan_count = 0
        for i in range(NUM_REPLAYS):
            input_buf.zero_()
            topk_ids.zero_()
            input_buf[0].copy_(torch.randn(HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10)
            topk_ids[0].copy_(torch.randint(0, NUM_EXPERTS, (TOPK,), device=h.device, dtype=torch.int64))

            graph.replay()
            torch.cuda.synchronize()

            real_out = graph_out[0:1]
            if real_out.isnan().any().item() or real_out.isinf().any().item():
                nan_count += 1
                if nan_count <= 3:
                    pad_nan = graph_out[1:].isnan().sum().item()
                    h.log_err(f"REPLAY {i}: real row NaN={real_out.isnan().sum().item()}, "
                              f"padding NaN={pad_nan}/{graph_out[1:].numel()}")
            elif i % 50 == 0:
                pad_nan = graph_out[1:].isnan().sum().item()
                h.log(f"  Replay {i}/{NUM_REPLAYS}: real OK, padding NaN={pad_nan}")

        h.log_all(f"Padded bs=1: {nan_count}/{NUM_REPLAYS} NaN in real rows")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
