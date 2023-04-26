# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from olla.dataflow_graph import Node, Edge, Graph
from typing import Iterable, Optional
import torch
from torch import fx
from olla.torch.fx_profiler import ProfilingInterpreter as Interpreter
from olla.training_graph_optimizer import Scheduler
import time


# class Profiler(Interpreter):
#     def __init__(
#         self,
#         gm: torch.fx.GraphModule,
#         profile_memory: bool = True,
#         profile_time: bool = False,
#         warm_up_iters: int = 0,
#         profile_iters: int = 1,
#     ):
#         super().__init__(gm)
#         self.profile_memory = profile_memory
#         self.profile_time = profile_time
#         self.warm_up_iters = warm_up_iters
#         self.profile_iters = profile_iters
#         if self.profile_memory:
#             assert (
#                 torch.cuda.is_available()
#             ), "Currently memory profile is only supported on CUDA"
#         self._reset()


class Simulator:
    def __init__(
        self,
        graph: Graph,
        model: torch.nn.Module,
        fx_2_df_node_map: dict[fx.Node, Node],
        *args,
        memory_bandwidth: int = int(1e11),
    ):
        self.graph: Graph = graph
        self.fx_2_df_node_map: dict[fx.Node, Node] = fx_2_df_node_map
        self.memory_bandwidth: int = memory_bandwidth
        self.intetrepreter: Interpreter = Interpreter(model, False, True)
        self.node_run_time: dict[Node, float] = {}

        self.intetrepreter.run(*args)

        for fx_node, df_node in self.fx_2_df_node_map.items():
            try:
                # print(fx_node.name)
                self.intetrepreter.node_profiles[fx_node]["runtimes_sec"]
                self.node_run_time[df_node] = sum(
                    self.intetrepreter.node_profiles[fx_node]["runtimes_sec"]
                ) / len(self.intetrepreter.node_profiles[fx_node]["runtimes_sec"])
            except Exception as e:
                # print(e)
                # print(fx_node.name)
                self.node_run_time[df_node] = 0
            # print(self.node_run_time[df_node])

    def _df_2_fx_node(self, node: Node) -> fx.Node:
        for fx_node, df_node in self.fx_2_df_node_map.items():
            if df_node == node:
                return fx_node
        raise ValueError(f"Node {node} not found in the graph")

    def _operator_time_estimation(self, node: Node) -> float:
        return self.node_run_time[node]

    def _data_transfer_time_estimation(self, edge: Edge) -> float:
        return edge.size / self.memory_bandwidth

    def default_simulate(self, node_ordering) -> float:
        total_time_cost: float = 0.0
        for node in node_ordering:
            total_time_cost += self.node_run_time[node]
        return total_time_cost

    def simulate_schedule(self) -> float:
        scheduler: Scheduler = Scheduler(self.graph)
        out = scheduler.ComputeOptimalSwapSchedule(
            mem_limit=int(1e9),
            eval_compute_cost=self.node_run_time,
            bandwidth=self.memory_bandwidth,
        )
        return self._simulate(*out)

    # def ComputeOptimalSwapSchedule(
    #     self, mem_limit, eval_compute_cost=None, bandwidth=1, defrag=False
    # ):

    def _simulate(
        self,
        node_ordering: Iterable[Node],
        swap_in_begin: list[Optional[Edge]],
        swap_in_end: list[Optional[Edge]],
        swap_out_begin: list[Optional[Edge]],
        swap_out_end: list[Optional[Edge]],
    ) -> tuple[int, list[tuple[Node, int]], float]:
        # edge_ref_counts = defaultdict(lambda: 0)

        # memory_used = self.graph.unused_weight_size
        # peak_memory: float = 0.0
        # mem_per_timestep: list[float] = []
        total_time_cost: float = 0.0
        time_per_timestep: list[float] = []

        tensor_swap_in_begin_time: dict[Edge, float] = {}
        tensor_swap_out_begin_time: dict[Edge, float] = {}

        # n: Node
        # fanout: Edge
        # for n in node_ordering:
        #     for fanout in n.fanout:
        #         # Record the number of references to this edge
        #         if fanout.size > 0:
        #             edge_ref_counts[fanout] = len(fanout.sinks)
        #             memory_used += fanout.size

        #     if memory_used > peak_memory:
        #         peak_memory = memory_used
        #     mem_per_timestep.append((n, memory_used))

        #     # Free memory with no more references
        #     for fanin in n.fanin:
        #         if fanin.size == 0:
        #             continue
        #         edge_ref_counts[fanin] -= 1
        #         assert edge_ref_counts[fanin] >= 0
        #         if edge_ref_counts[fanin] == 0:
        #             memory_used -= fanin.size

        for (
            i,
            node,
            edge_in_begin,
            edge_in_end,
            edge_out_begin,
            edge_out_end,
        ) in enumerate(
            zip(node_ordering, swap_in_begin, swap_in_end, swap_out_begin, swap_out_end)
        ):
            # Compute the time cost of the current timestep
            node_time: float = self._operator_time_estimation(node)
            node_end_time: float = total_time_cost + node_time
            edge_in_end_time: float = total_time_cost
            edge_out_end_time: float = total_time_cost

            # Assume no swap is None
            if edge_in_begin is not None:
                tensor_swap_in_begin_time[edge_in_begin] = total_time_cost

            if edge_out_begin is not None:
                tensor_swap_out_begin_time[edge_out_begin] = total_time_cost

            if edge_in_end is not None:
                edge_time: float = self._data_transfer_time_estimation(edge_in_end)
                assert (
                    edge_in_end in tensor_swap_in_begin_time
                ), "Edge not in tensor_swap_in_begin_time"
                edge_in_end_time = tensor_swap_in_begin_time[edge_in_end] + edge_time

            if edge_out_end is not None:
                edge_time: float = self._data_transfer_time_estimation(edge_out_end)
                assert (
                    edge_out_end in tensor_swap_out_begin_time
                ), "Edge not in tensor_swap_out_begin_time"
                edge_out_end_time = tensor_swap_out_begin_time[edge_out_end] + edge_time

            time_per_timestep[i] = (
                max(node_end_time, edge_in_end_time, edge_out_end_time)
                - total_time_cost
            )
            total_time_cost += time_per_timestep[i]

        return total_time_cost, time_per_timestep
