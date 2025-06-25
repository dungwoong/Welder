import functools
import math
from queue import PriorityQueue
from typing import Dict, Generator, Iterable, List

import numpy as np
import tvm

from ..arch import Arch
from ..bestfit import BestFit
from ..config import Config, Stride, TileDict
from ..graph import IRNode, Node, find_topo_sort
from .common import (coalesced_factor, coalesced_tensor_shape, factorize,
                     get_all_factors)
from ..rasterization import NoRasterization

class DefaultPolicy:
    def __init__(self, output_nodes: List[Node], arch:Arch) -> None:
        self.arch = arch
        self.ordered_nodes = list(filter(
            lambda n: not n.is_placeholder() and not n.is_output(),
            find_topo_sort(output_nodes)
        ))

        self.output_nodes = []
        for node in self.ordered_nodes:
            is_topo_output = True
            for edge in node.outputs:
                if not edge.dst_node.is_output():
                    is_topo_output = False
            if is_topo_output:
                self.output_nodes.append(node)

    def emit_config(self, topk: int) -> List[Dict[Node, Config]]:
        """
        Emits a config for each node, so it's a dict like {node: {config of tile sizes}}

        At each assignment, it has a scoring system so we can figure that out btw

        This is just for a singular node: {'block': [128, 128], 'warp': [64, 64], 'wmma': [16, 8, 16], 'use_cutlass': True, 'rstep': [32], 'use_tc': '86', 'strides': {2: <Stride, 0, 136>}}
        - reduction step is how much we're gonna reduce over per loop(eg. for gemm, rstep 32 means k_tile is 32)

        A single smem_tile_condidate looks like this:
        - rstep_map = {node: {'k': 32}}
        - smem_cost (32768), output_tile[128, 128]
        - traffic: 2129920.0
        - grid_size: 1024
        - num_wave: 8
        - use_cutlass_mma: {node: True}]

        Wait so maybe they ARE actually just planning out SMEM tiles right now
        """
        try:
            base_tile = self.get_base_tile()
        except Exception:
            return []
        if base_tile is None:
            return []
        rstep_map = {node : self._assign_reduce_step(node) for node in self.ordered_nodes}
        smem_tile_condidates = self.DFS_smem_tile(base_tile, topk, rstep_map)
        results = []
        for td in smem_tile_condidates:
            if not self.check_tile_shape_isvalid(td):
                continue
            block_orders = self._assign_block_order(td)
            if block_orders is False:
                continue
            self._expand_reduce_axis(td)
            rasterization = self.plan_rasterization(td)
            # THIS is where they plan the warp memory level now
            for codegen_dicts in self.assign_block_size(td): # generating yielding dicts like {block:... warp:..., rstep: [...]}. THIS IS WHERE they plan warp tiling
                # handle cases where block is not ordinal (e.g. transpose)
                for node, block_order in block_orders.items():
                    codegen_dicts[node].block_order = block_order
                for node, strides in td.output_strides_map.items():
                    codegen_dicts[node].output_strides = strides
                codegen_dicts["globals"]["Rasterization"] = rasterization
                results.append(codegen_dicts)
                if len(results) >= topk:break
            if len(results) >= topk:break
        return results # 20 results, each is a codegen dict.

    def DFS_smem_tile(self, init_tile, topk, rstep_map) -> Iterable[TileDict]:
        _steps = [get_all_factors(n) for n in self.output_nodes[0].get_space_dim()]
        steps = [step[step.index(t):] for step, t in zip(_steps, init_tile)]
        for i in range(len(steps)):
            added = list(filter(lambda s:s < steps[i][-1] and s > steps[i][0] and s not in steps[i], [2, 4, 8, 16, 32]))
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tiles = {}
        queue = PriorityQueue()
        def prio(td: TileDict):
            return (td.traffic + 1) * td.num_wave # * (td.block_per_SM ** 0.5)
        def add_to_queue(tile):
            if tuple(tile) in visited_tiles:
                return
            td = self.compute_tile_dict(tile, rstep_map)
            visited_tiles[tuple(tile)] = td
            if td.valid:
                queue.put([prio(td), tile])

        add_to_queue(init_tile)
        while not (queue.empty() or len(visited_tiles) > 2000):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tiles = filter(lambda td: td.valid, visited_tiles.values())
        sorted_tiles = sorted(visited_tiles, key=lambda td:prio(td))
        return sorted_tiles

    # get the minimum tile that could satisfy no redundancy computation
    def get_base_tile(self):
        if len(set([len(node.get_space_dim()) for node in self.output_nodes])) > 1:
            # If output dim sizes are not same, don't know how to handle them
            return None
        out_node = self.output_nodes[0]
        shape = out_node.get_space_dim()
        base_tile = [1 for _ in shape]
        wpi = self.compute_workload_per_item(base_tile)
        for dim, n in enumerate(shape):
            factors = [n]
            for factor in factors:
                if factor == base_tile[dim]:continue
                tile = base_tile.copy()
                tile[dim] = factor
                new_wpi = self.compute_workload_per_item(tile)
                if new_wpi < wpi:
                    wpi, base_tile = new_wpi, tile
                else:
                    break

        return base_tile

    # handles multiple output cases
    def _get_output_tile_map(self, tile):
        tile_map = {}
        for node in self.output_nodes:
            tile_map[node] = [
                tile[i] * node.get_space_dim()[i] // self.output_nodes[0].get_space_dim()[i] for i in range(len(tile))]
        return tile_map

    def compute_workload_per_item(self, output_tile) -> float:
        op_tile_map = self._get_output_tile_map(output_tile)
        compute = 0
        num_item = int(np.prod(output_tile))
        for node in reversed(self.ordered_nodes):
            tile = op_tile_map[node]
            dep = node.propogate_inputs(tile)
            compute += int(np.prod(tile))
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node] = dep[i]
        return float(compute / num_item)

    def score_block_size(self, n):
        num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
        r1 = max(num_wrap/self.arch.sm_partition, self.arch.sm_partition/num_wrap)
        r2 = (num_wrap * self.arch.warp_size - n) / n
        return (r1, r2)

    def get_block_size(self, n):
        factors = get_all_factors(n)
        factors = list(filter(lambda x: x <= 1024, factors))
        factor_ordered = sorted(factors, key=self.score_block_size)
        return factor_ordered[0]

    def get_node_reduce_step_candidates(self, node):
        # general idea : use factor first, since it does not require extra boundary check
        #                for large prime number, which is rare case, use power of 2.
        results = {}
        for k in node.raxis:
            all_factors = get_all_factors(node.raxis[k])
            if len(all_factors) == 2 and node.raxis[k] > 64:
                all_factors = [1]
                while all_factors[-1] * 2 < node.raxis[k]:
                    all_factors.append(all_factors[-1] * 2)
            results[k] = all_factors
        return results

    def _assign_reduce_step(self, node: IRNode):
        if len(node.raxis) == 0:
            return {}
        raxis = node.raxis
        tile = [1] * len(node.get_space_dim())
        all_steps = self.get_node_reduce_step_candidates(node)
        def sim(a, b):
            return  (2 * a * b) / (a * a + b * b)

        def _score(rstep_id):
            rstep = {k : all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = node.propogate_inputs(tile, rstep=rstep)
            for edge in node.inputs:
                if edge.src_node.is_placeholder():
                    read_transaction_elements = self.arch.transaction_size[1] // ((edge.src_node.get_dtype().bits + 7) // 8)
                    score += sim(coalesced_factor(shape[edge.dst_id], edge.src_node.get_shape()), read_transaction_elements)
            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x:x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_rstep_id = {ax : 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)
        while True:
            if cur_score == 0:break
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            else:
                cur_rstep_id, cur_score = new_rstep, new_score
        rstep = {k : all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _expand_reduce_axis(self, td: TileDict):
        smem_limit = min(self.arch.max_smem_usage // td.block_per_SM, self.arch.smem_cap)
        rstep_map = td.rstep_map.copy()
        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)
            for k in all_steps:
                all_steps[k] = list(filter(lambda x: x % rstep[k] == 0, all_steps[k]))

            def _score(rstep_id):
                rstep = {k : all_steps[k][rstep_id[k]] for k in node.raxis}
                score = 0
                shape = node.propogate_inputs(td.get_tile(node), rstep=rstep)
                for edge in node.inputs:
                    if edge.src_node.is_placeholder():
                        factor = coalesced_factor(shape[edge.dst_id], edge.src_node.get_shape())
                        score += factor
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return None
                return max(candidates, key=lambda x:x[1])[0]

            cur_rstep_id = {k : all_steps[k].index(rstep[k]) for k in node.raxis}
            new_rstep_map = rstep_map.copy()
            while True:
                new_rstep_id = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map[node] = {k : all_steps[k][new_rstep_id[k]] for k in node.raxis}
                old_rstep_map = td.rstep_map
                td.rstep_map = new_rstep_map
                smem_usage, _ = self._compute_shared_memory_usage(td)
                td.rstep_map = old_rstep_map
                if smem_usage > smem_limit:
                    break
                else:
                    cur_rstep_id = new_rstep_id
            rstep = {k : all_steps[k][cur_rstep_id[k]] for k in node.raxis}
            return rstep

        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)

    def _compute_memory_traffic(self, output_tile):
        op_tile_map = self._get_output_tile_map(output_tile)
        traffic = 0
        for node in reversed(self.ordered_nodes):
            tile = op_tile_map[node]
            input_shapes = node.propogate_inputs(tile)
            output_shapes = node.propogate_outputs(tile)
            for i, edge in enumerate(node.inputs):
                op_tile_map[edge.src_node] = input_shapes[i]
                if edge.src_node.is_placeholder():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    read_transaction_elements = self.arch.transaction_size[1] // nbytes
                    traffic += coalesced_tensor_shape(input_shapes[i], edge.src_node.get_shape(), read_transaction_elements) * nbytes
            for edge in node.outputs:
                if edge.dst_node.is_output():
                    nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                    write_transaction_elements = self.arch.transaction_size[0] // nbytes
                    traffic += coalesced_tensor_shape(output_shapes[edge.src_id], node.get_shape(edge.src_id), write_transaction_elements) * nbytes
        return traffic, op_tile_map

    def infer_node_smem_usage(self, td: TileDict, node: IRNode):
        return node.footprint(td.get_tile(node), td.get_rstep(node), td.tensor_strides_map[node])

    def _compute_shared_memory_usage(self, td: TileDict):
        self._compute_stride_map(td)
        allocator = BestFit()
        block_map = {}
        processed = set()
        cached_tensors_map = {}
        def can_free(node, out_id):
            for edge in node.outputs:
                if edge.src_id == out_id and edge.dst_node not in processed:
                    return False
            return True
        for node in self.ordered_nodes:
            node_internal_bytes, cached_tensors_map[node] = self.infer_node_smem_usage(td, node)
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                    allocator.free(block_map.pop((edge.src_node, edge.src_id)))
            # alloc outputs
            for edge in node.outputs:
                if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                    dtype_bytes = (node.get_dtype(edge.src_id).bits + 7) // 8
                    stride = td.output_strides_map[node][len(node.inputs) + edge.src_id]
                    output_elem = stride.compute_elements_from_shape(td.get_tile(node))
                    block_map[(node, edge.src_id)] = allocator.malloc(output_elem * dtype_bytes)

        assert len(block_map) == 0
        return allocator.limit, cached_tensors_map

    def compute_node_stride_map(self, node: IRNode, td: TileDict):
        output_strides = {int(edge.src_id + len(node.inputs)): Stride() for edge in node.outputs}
        tensor_strides = {}
        return output_strides, tensor_strides

    def _compute_stride_map(self, td: TileDict):
        output_strides_map = {}
        tensor_strides_map = {}
        for node in self.ordered_nodes:
            output_strides_map[node], tensor_strides_map[node] = self.compute_node_stride_map(node, td)
            for name, stride in tensor_strides_map[node].items():
                arg_names = [arg.name for arg in node.args]
                if name in arg_names:
                    input_id = arg_names.index(name)
                else:
                    continue  # don't need to propogate internal strides
                src_id, src_node = node.inputs[input_id].src_id, node.inputs[input_id].src_node
                if not src_node.is_placeholder():
                    output_strides_map[src_node][int(src_id + len(src_node.inputs))] = stride

        td.output_strides_map, td.tensor_strides_map = output_strides_map, tensor_strides_map

    def compute_tile_dict(self, output_tile: List[int], rstep_map) -> TileDict:
        td = TileDict(output_tile)
        td.rstep_map = rstep_map
        td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
        if td.smem_cost > self.arch.smem_cap:
            td.valid = False
            return td
        output_shape = self.output_nodes[0].get_space_dim()
        td.grid_size = int(np.prod([(y + x - 1) // x for x, y in zip(output_tile, output_shape)]))
        # estimated reg usage
        reg_usage = int(2 * max([np.prod(td.get_tile(node)) * node.get_dtype().bits / 32 for node in self.ordered_nodes]))
        if reg_usage > self.arch.reg_cap:
            td.valid = False
            return td
        td.block_per_SM = min(self.arch.max_smem_usage // max(td.smem_cost, 1), self.arch.reg_cap // max(reg_usage, 1), self.arch.sm_partition)
        td.num_wave = int(np.ceil(td.grid_size / (td.block_per_SM * self.arch.compute_max_core)))
        return td

    def check_tile_shape_isvalid(self, td: TileDict):
        for node in self.ordered_nodes:
            if np.prod(td.get_tile(node)) == 0: return False
            node_grid_size = np.prod([(y + x - 1) // x for x, y in zip(td.get_tile(node), node.get_space_dim())])
            if node_grid_size != td.grid_size:
                return False
        return True

    def recommend_block_size(self, td: TileDict) -> List[int]:
        node_space_sizes = [int(np.prod(td.get_tile(node))) for node in self.ordered_nodes]
        max_block_size = functools.reduce(math.gcd, node_space_sizes)

        if max_block_size < self.arch.warp_size * self.arch.sm_partition and max_block_size == min(node_space_sizes):
            node_reduce_sizes = [int(np.prod(list(td.get_rstep(node).values()))) for node in self.ordered_nodes]
            total_sizes = [x * y for x, y in zip(node_space_sizes, node_reduce_sizes)]
            max_possible_size = functools.reduce(math.gcd, total_sizes)
            possible_block_sizes = list(filter(
                lambda x: x % max_block_size == 0 and x <= 1024, get_all_factors(max_possible_size)))
            possible_block_sizes = list(filter( # either be a factor of space or cover fully cover the space
                lambda x: all([x % s == 0 or s % x == 0 for s in node_space_sizes]) , possible_block_sizes))
            factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
            return factor_ordered
        else:
            possible_block_sizes = get_all_factors(max_block_size)
            possible_block_sizes = list(filter(lambda x: x <= 1024, possible_block_sizes))
        factor_ordered = sorted(possible_block_sizes, key=self.score_block_size)
        return factor_ordered

    def assign_block_size(self, td: TileDict, topk=1) -> Generator[Dict, Node, Config]:
        block_size_ordered = self.recommend_block_size(td)
        for block_size in block_size_ordered:
            result = {}
            result["globals"] = {}
            failed = False
            for node in self.ordered_nodes:
                result[node] = self._assign_block_size(node, td, block_size)
                if result[node] is None:
                    failed = True
                    break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_block_order(self, td: TileDict):
        block_idx = tvm.te.var("block_idx")
        analyzer = tvm.arith.Analyzer()
        analyzer.update(block_idx, tvm.arith.ConstIntBound(0, td.grid_size - 1))
        expr_map = {node : block_idx for node in self.output_nodes}
        result = {}
        for node in reversed(self.ordered_nodes):
            expr = expr_map[node]
            if not (expr.same_as(block_idx) or isinstance(expr, tvm.tir.expr.ConstExpr)):
                result[node] = expr
            deps = node.block_infer(td.tile_map, expr)
            for i, edge in enumerate(node.inputs):
                if edge.src_node.is_placeholder(): continue
                deps[i] = analyzer.simplify(deps[i])
                if edge.src_node in expr_map:
                    if not deps[i].same_as(expr_map[edge.src_node]):
                        return False
                else:
                    expr_map[edge.src_node] = deps[i]
        return result

    def _assign_block_size(self, node: IRNode, td: TileDict, block_size: int):
        tile, rsteps = td.get_tile(node), td.get_rstep(node)
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        reduce_thread = {k : 1 for k in rsteps}
        ndim = len(tile)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.propogate_inputs(block_tile)
            for edge in node.inputs:
                score += np.prod(shape[edge.dst_id]) / self.arch.bandwidth[1]
            for edge in node.outputs:
                if edge.dst_node.is_output(): # write to global
                    score += coalesced_tensor_shape(thread, node.get_shape(), 8) / self.arch.bandwidth[0]
            return score
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                if (tile[i] % (cur_threads[i] * factor)) != 0:
                    continue
                cur_threads[i] *= factor
                score_map[i] = (_score(node, cur_threads), i)
                cur_threads[i] //= factor
            if len(score_map) > 0:
                # assign to space axis
                dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
                cur_threads[dim_order[0]] *= factor
            else:
                # assign to reduce axis
                target_ax = None
                for ax, ax_len in reversed(list(rsteps.items())):
                    if ax_len % (reduce_thread[ax] * factor) == 0:
                        target_ax = ax
                        break
                assert target_ax
                reduce_thread[target_ax] *= factor

        codegen_dict = Config()
        codegen_dict.block = tile
        codegen_dict.thread = cur_threads
        codegen_dict.rstep = [rsteps[ax] for ax in node.raxis]
        codegen_dict.reduce_thread = [reduce_thread[ax] for ax in node.raxis]
        codegen_dict.cached_tensors = td.cached_tensors_map[node]
        codegen_dict.schedule_stages = [stage.name for stage in node.schedule_stages]
        if node.get_dtype().bits == 16: # set step=2 for fp16 case
            codegen_dict._step = [1 for _ in range(ndim)]
            for i in reversed(range(ndim)):
                if codegen_dict.block[i] // codegen_dict.thread[i] % 2 == 0:
                    codegen_dict._step[i] = 2
                    break
        # Plan vectorize
        codegen_dict.vectorize = self._plan_vectorize(node, td, block_size)
        return codegen_dict

    def _plan_vectorize(self, node: IRNode, td: TileDict, block_size: int):
        def is_cont(shape, vec):
            if len(shape) == 0: return vec == 1
            last = shape[-1]
            if last == 1:
                return is_cont(shape[0:-1], vec // last)
            else:
                return last % vec == 0
        def is_shape_aligned(shape, factor):
            return int(np.prod(shape)) % factor == 0
        def is_type_allowed(dtype, vec):
            return dtype.bits * vec <= 128
        vectorize_sizes = [8, 4, 2]
        dtypes = node.get_reduce_inputs_dtype()
        shapes = node.propogate_reduction_inputs(td.get_tile(node), td.get_rstep(node))
        vectorize_result = {}
        for tensor, shape in shapes.items():
            for v in vectorize_sizes:
                if is_shape_aligned(shape, block_size * v) and is_cont(shape, v) and is_type_allowed(dtypes[tensor], v):
                    vectorize_result[tensor] = v
                    break
        return vectorize_result

    def plan_rasterization(self, td: TileDict):
        return NoRasterization()
