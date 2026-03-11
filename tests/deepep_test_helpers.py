"""
Shared setup for DeepEP CUDA graph tests.
Not run directly — imported by test_*.py scripts.
"""
import os
import sys

import torch
import torch.distributed

import deep_ep

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
    DeepEPLLPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernel
from vllm.v1.worker.workspace import init_workspace_manager

# DeepSeek-R1 config
NUM_EXPERTS = 256
TOPK = 1
HIDDEN_SIZE = 7168
INTERMEDIATE_SIZE = 2048
MAX_TOKENS_PER_RANK = 64
NUM_REPLAYS = 200


def make_dummy_moe_config():
    return FusedMoEConfig(
        num_experts=1,
        experts_per_token=1,
        hidden_dim=1,
        intermediate_size_per_partition=1,
        num_local_experts=1,
        num_logical_experts=1,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.TopK,
    )


def make_deepep_ll_a2a(pg, world_size, num_experts, max_tokens_per_rank,
                        hidden_size, use_fp8_dispatch=False):
    num_local_experts = num_experts // world_size
    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        max_tokens_per_rank, hidden_size, world_size, num_experts,
    )
    buffer = deep_ep.Buffer(
        group=pg,
        num_rdma_bytes=num_rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=num_local_experts,
    )
    return DeepEPLLPrepareAndFinalize(
        buffer=buffer,
        num_dispatchers=world_size,
        max_tokens_per_rank=max_tokens_per_rank,
        use_fp8_dispatch=use_fp8_dispatch,
    )


def make_weights(num_experts, n, k, dtype=torch.bfloat16):
    w1 = torch.randn((num_experts, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((num_experts, k, n), device="cuda", dtype=dtype) / 10
    return w1, w2


class DeepEPTestHarness:
    """Sets up distributed env, weights, expert map, and MoE kernel."""

    def __init__(self, max_tokens_per_rank=MAX_TOKENS_PER_RANK):
        self.max_tokens_per_rank = max_tokens_per_rank
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % 4))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        init_workspace_manager(self.device)

        torch.distributed.init_process_group(backend="nccl", world_size=self.world_size)
        self.pg = torch.distributed.new_group(list(range(self.world_size)))

        self.num_local_experts = NUM_EXPERTS // self.world_size

        self.w1, self.w2 = make_weights(self.num_local_experts, INTERMEDIATE_SIZE, HIDDEN_SIZE)

        self.expert_map = torch.full((NUM_EXPERTS,), fill_value=-1, dtype=torch.int32)
        e_start = self.rank * self.num_local_experts
        e_end = e_start + self.num_local_experts
        self.expert_map[e_start:e_end] = torch.arange(self.num_local_experts, dtype=torch.int32)
        self.expert_map = self.expert_map.to(device=self.device)

        a2a = make_deepep_ll_a2a(
            pg=self.pg, world_size=self.world_size,
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=self.max_tokens_per_rank,
            hidden_size=HIDDEN_SIZE,
            use_fp8_dispatch=True,
        )

        quant_config = FusedMoEQuantConfig.make(None)
        moe_config = make_dummy_moe_config()

        fused_experts = BatchedTritonExperts(
            max_num_tokens=self.max_tokens_per_rank,
            num_dispatchers=self.world_size,
            moe_config=moe_config,
            quant_config=quant_config,
        )

        self.mk = FusedMoEKernel(
            prepare_finalize=a2a,
            fused_experts=fused_experts,
            inplace=False,
        )

        self.vllm_ctx = set_current_vllm_config(VllmConfig())

    def moe_forward(self, hidden_states, topk_ids, topk_weights):
        return self.mk.apply(
            hidden_states=hidden_states,
            w1=self.w1, w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=NUM_EXPERTS,
            expert_map=self.expert_map,
            apply_router_weight_on_input=False,
        )

    def warmup(self, input_buf, topk_ids_buf, topk_weights_buf, n=3):
        for _ in range(n):
            out = self.moe_forward(input_buf, topk_ids_buf, topk_weights_buf)
        torch.cuda.synchronize()
        return out

    def capture_graph(self, input_buf, topk_ids_buf, topk_weights_buf):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = self.moe_forward(input_buf, topk_ids_buf, topk_weights_buf)
        torch.cuda.synchronize()
        return graph, graph_out

    def log(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def log_all(self, msg):
        print(f"[RANK {self.rank}] {msg}", flush=True)

    def log_err(self, msg):
        print(f"[RANK {self.rank}] {msg}", file=sys.stderr, flush=True)

    def teardown(self):
        torch.distributed.destroy_process_group()


class ChainedMoEHarness:
    """Sets up N MoE layers sharing a single deep_ep.Buffer, like real DeepSeek-R1."""

    def __init__(self, num_layers, max_tokens_per_rank=MAX_TOKENS_PER_RANK):
        self.num_layers = num_layers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank % 4))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        init_workspace_manager(self.device)

        torch.distributed.init_process_group(backend="nccl", world_size=self.world_size)
        self.pg = torch.distributed.new_group(list(range(self.world_size)))

        self.num_local_experts = NUM_EXPERTS // self.world_size

        self.expert_map = torch.full((NUM_EXPERTS,), fill_value=-1, dtype=torch.int32)
        e_start = self.rank * self.num_local_experts
        e_end = e_start + self.num_local_experts
        self.expert_map[e_start:e_end] = torch.arange(self.num_local_experts, dtype=torch.int32)
        self.expert_map = self.expert_map.to(device=self.device)

        # Shared buffer across all layers (like real model)
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            max_tokens_per_rank, HIDDEN_SIZE, self.world_size, NUM_EXPERTS,
        )
        self.shared_buffer = deep_ep.Buffer(
            group=self.pg,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=self.num_local_experts,
        )

        quant_config = FusedMoEQuantConfig.make(None)
        moe_config = make_dummy_moe_config()

        # Each layer gets its own a2a, experts, weights, and kernel
        self.layers = []
        self.all_w1 = []
        self.all_w2 = []
        for _ in range(num_layers):
            a2a = DeepEPLLPrepareAndFinalize(
                buffer=self.shared_buffer,
                num_dispatchers=self.world_size,
                max_tokens_per_rank=max_tokens_per_rank,
                use_fp8_dispatch=True,
            )
            experts = BatchedTritonExperts(
                max_num_tokens=max_tokens_per_rank,
                num_dispatchers=self.world_size,
                moe_config=moe_config,
                quant_config=quant_config,
            )
            mk = FusedMoEKernel(
                prepare_finalize=a2a,
                fused_experts=experts,
                inplace=False,
            )
            w1, w2 = make_weights(self.num_local_experts, INTERMEDIATE_SIZE, HIDDEN_SIZE)
            self.layers.append(mk)
            self.all_w1.append(w1)
            self.all_w2.append(w2)

        self.vllm_ctx = set_current_vllm_config(VllmConfig())

    def chain_forward(self, hidden_states, topk_ids, topk_weights):
        """Run all layers sequentially, each layer's output -> next layer's input."""
        x = hidden_states
        for i, mk in enumerate(self.layers):
            x = mk.apply(
                hidden_states=x,
                w1=self.all_w1[i], w2=self.all_w2[i],
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=NUM_EXPERTS,
                expert_map=self.expert_map,
                apply_router_weight_on_input=False,
            )
        return x

    def warmup(self, input_buf, topk_ids_buf, topk_weights_buf, n=3):
        for _ in range(n):
            out = self.chain_forward(input_buf, topk_ids_buf, topk_weights_buf)
        torch.cuda.synchronize()
        return out

    def capture_graph(self, input_buf, topk_ids_buf, topk_weights_buf):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = self.chain_forward(input_buf, topk_ids_buf, topk_weights_buf)
        torch.cuda.synchronize()
        return graph, graph_out

    def log(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def log_all(self, msg):
        print(f"[RANK {self.rank}] {msg}", flush=True)

    def log_err(self, msg):
        print(f"[RANK {self.rank}] {msg}", file=sys.stderr, flush=True)

    def teardown(self):
        torch.distributed.destroy_process_group()
