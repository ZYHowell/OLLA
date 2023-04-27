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
        memory_bandwidth: int = int(1e8),
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
        # print(f"len(node_ordering): {len(node_ordering)}")
        for node in node_ordering:
            total_time_cost += self.node_run_time[node]
        return total_time_cost

    def simulate_schedule(self) -> float:
        scheduler: Scheduler = Scheduler(self.graph)
        out = scheduler.ComputeOptimalSwapSchedule(
            mem_limit=int(3.1e8),
            eval_compute_cost=self.node_run_time,
            bandwidth=self.memory_bandwidth,
        )
        # print(out)
        return self._simulate(*out)

    def _simulate(
        self,
        node_ordering: dict[int, Node],
        swap_in_begin: dict[int, Optional[Edge]],
        swap_in_end: dict[int, Optional[Edge]],
        swap_out_begin: dict[int, Optional[Edge]],
        swap_out_end: dict[int, Optional[Edge]],
    ) -> float:
        total_time_cost: float = 0.0
        # print(swap_in_begin)

        tensor_swap_in_begin_time: dict[Edge, float] = {}
        tensor_swap_out_begin_time: dict[Edge, float] = {}

        # print(f"ts: {len(node_ordering)}")
        for ts in node_ordering.keys():
            # print(ts)
            node = node_ordering[ts]
            edge_in_begin = swap_in_begin[ts]
            edge_in_end = swap_in_end[ts]
            edge_out_begin = swap_out_begin[ts]
            edge_out_end = swap_out_end[ts]

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
                # print(edge_time)
                assert (
                    edge_in_end in tensor_swap_in_begin_time
                ), "Edge not in tensor_swap_in_begin_time"
                edge_in_end_time = tensor_swap_in_begin_time[edge_in_end] + edge_time
                # print(edge_in_end_time)

            if edge_out_end is not None:
                edge_time: float = self._data_transfer_time_estimation(edge_out_end)
                # print(edge_time)
                assert (
                    edge_out_end in tensor_swap_out_begin_time
                ), "Edge not in tensor_swap_out_begin_time"
                edge_out_end_time = tensor_swap_out_begin_time[edge_out_end] + edge_time

            total_time_cost = max(node_end_time, edge_in_end_time, edge_out_end_time)
            # total_time_cost = node_end_time
            # print(f"total_time_cost: {total_time_cost}")

        return total_time_cost
