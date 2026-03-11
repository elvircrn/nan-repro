#!/usr/bin/env python3
"""
Test 5: Graph captured at MAX, each rank gets a different effective_bs.
Simulates real P/D where ranks have unequal load — the NVSHMEM all-to-all
sees mismatched token counts across ranks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from deepep_test_helpers import *


def main():
    h = DeepEPTestHarness()
    capture_bs = MAX_TOKENS_PER_RANK

    with h.vllm_ctx:
        h.log(f"\n=== Asymmetric per-rank bs (captured at {capture_bs}) ===")

        input_buf = torch.randn(capture_bs, HIDDEN_SIZE, device=h.device, dtype=torch.bfloat16) / 10
        topk_ids = torch.randint(0, NUM_EXPERTS, (capture_bs, TOPK), device=h.device, dtype=torch.int64)
        topk_w = torch.ones(capture_bs, TOPK, device=h.device, dtype=torch.float32)

        h.warmup(input_buf, topk_ids, topk_w)
        graph, graph_out = h.capture_graph(input_buf, topk_ids, topk_w)
        h.log("Graph captured.")

        nan_count = 0
        for i in range(NUM_REPLAYS):
            # Each rank gets a different effective batch size
            if h.rank == 0:
                effective_bs = 1
            else:
                effective_bs = (h.rank * 7 + i * 3) % capture_bs + 1

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
                h.log(f"  Replay {i}/{NUM_REPLAYS}: rank0_bs=1 OK")

        h.log_all(f"Asymmetric: {nan_count}/{NUM_REPLAYS} NaN")
        if nan_count > 0:
            h.teardown()
            sys.exit(1)

    h.log_all("PASS")
    h.teardown()


if __name__ == "__main__":
    main()
