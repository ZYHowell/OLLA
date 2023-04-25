# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from olla.dataflow_graph import Node, Edge, Graph
from typing import Iterable, Optional


class Simulator:
    def __init__(self, graph: Graph, memory_bandwidth: float = 0):
        self.graph: Graph = graph
        self.memory_bandwidth: float = memory_bandwidth

    def _operator_time_estimation(self, node: Node) -> float:
        return 0.0

    def _data_transfer_time_estimation(self, edge: Edge) -> float:
        return edge.size / self.memory_bandwidth

    def Simulate(self, 
                 node_ordering: Iterable[Node], 
                 swap_in_begin: list[Optional[Edge]], 
                 swap_in_end: list[Optional[Edge]], 
                 swap_out_begin: list[Optional[Edge]], 
                 swap_out_end: list[Optional[Edge]]
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

        for i, node, edge_in_begin, edge_in_end, edge_out_begin, edge_out_end in \
            enumerate(zip(node_ordering, swap_in_begin, swap_in_end, swap_out_begin, swap_out_end)):
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
                assert edge_in_end in tensor_swap_in_begin_time, "Edge not in tensor_swap_in_begin_time"
                edge_in_end_time = tensor_swap_in_begin_time[edge_in_end] + edge_time

            if edge_out_end is not None:
                edge_time: float = self._data_transfer_time_estimation(edge_out_end)
                assert edge_out_end in tensor_swap_out_begin_time, "Edge not in tensor_swap_out_begin_time"
                edge_out_end_time = tensor_swap_out_begin_time[edge_out_end] + edge_time

            time_per_timestep[i] = max(node_end_time, edge_in_end_time, edge_out_end_time) - total_time_cost
            total_time_cost += time_per_timestep[i]
            

        return total_time_cost, time_per_timestep
