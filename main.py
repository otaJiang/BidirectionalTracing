import networkx as nx
import hashlib
import json
from mpt.mpt import MerklePatriciaTrie
from mpt.hash import keccak_hash
from mpt.nibble_path import NibblePath
from mpt.node import Node
import binascii
import sys
import random
import time
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


def hash_data(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

def hash_combine(hashes):
    combined = ''.join(hashes)
    return hashlib.sha256(combined.encode()).hexdigest()

def read_graph_from_file(filename):
    G = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            dst, src = line.strip().split('\t')
            G.add_edge(src, dst)
    return G

def remove_cycles_dfs(G):
    """
    使用深度优先搜索（DFS）移除图中的所有环，返回一个有向无环图（DAG）。
    同时记录需要连接的前驱和后继节点。
    """
    visited = set()
    stack = set()
    edges_to_remove = []
    predecessors = {}
    successors = {}

    def dfs(u):
        visited.add(u)
        stack.add(u)
        for v in G.successors(u):
            if v not in visited:
                dfs(v)
            elif v in stack:
                # 检测到环，移除边 u -> v
                edges_to_remove.append((u, v))
                # 记录环的前驱和后继节点
                if u not in predecessors:
                    predecessors[u] = set()
                if v not in successors:
                    successors[v] = set()
                for pred in G.predecessors(u):
                    predecessors[u].add(pred)
                for succ in G.successors(v):
                    successors[v].add(succ)
        stack.remove(u)

    for node in G.nodes():
        if node not in visited:
            dfs(node)

    # 移除导致环的边
    G.remove_edges_from(edges_to_remove)

    return G, predecessors, successors

def connect_without_cycles(G, predecessors, successors):
    """
    将环的前驱节点连接到后继节点，确保不引入新的环。
    """
    # 获取拓扑排序
    topo_order = list(nx.topological_sort(G))
    # 创建节点的拓扑排序位置映射
    topo_position = {node: i for i, node in enumerate(topo_order)}

    # 遍历记录的前驱和后继节点
    for node in predecessors:
        preds = predecessors[node]
        succs = successors.get(node, set())
        for pred in preds:
            for succ in succs:
                if pred != succ and topo_position[pred] < topo_position[succ]:
                    # 添加边并确保不形成环
                    G.add_edge(pred, succ)
    return G

def compute_traces(G, nodes=None):
    """
    计算图中节点的 trace 值，如果提供了 nodes 列表，则只计算这些节点的 trace。
    """
    if nodes is None:
        nodes = list(G.nodes())
    topo_sort = list(nx.topological_sort(G))
    trace_map = {}
    time_trace={}
    vo_size_map = {}

    for node in topo_sort:
        if node not in nodes:
            continue

        # 如果是复制节点，找到原本节点
        start_time = time.perf_counter()
        vo_elements = []
        if "_replica" in str(node):
            original_node = node.replace("_replica", "")
            trace_map[node] = trace_map.get(original_node, hash_data(original_node))
        else:
            if G.in_degree(node) == 0:
                # 如果是起始节点，直接计算 hash 并记录 VO
                hash_value = hash_data(node)
                trace_map[node] = hash_value
                vo_elements.append(hash_value)  # 起始节点的 hash 作为 VO
            else:
                # 否则，计算前驱节点的 trace
                direct_precursors = list(G.predecessors(node))
                precursor_traces = [trace_map[precursor] for precursor in direct_precursors]
                trace_map[node] = hash_combine(precursor_traces)

                # 收集每个前驱节点链条上的起始节点哈希
                for precursor in direct_precursors:
                    if G.in_degree(precursor) == 0:
                        vo_elements.append(trace_map[precursor])

        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1_000_000
        time_trace[node] = elapsed_us

        # 计算 VO 的总大小
        vo_size = sum(sys.getsizeof(el) for el in vo_elements)
        vo_size_map[node] = vo_size

    return trace_map, time_trace, vo_size_map

def get_all_successors(G, node, memo=None):
    """
    获取当前节点的所有后驱节点，包括直接和间接的，使用缓存优化。
    """
    if memo is None:
        memo = {}

    if node in memo:
        return memo[node]

    all_successors = set()
    for successor in G.successors(node):
        all_successors.add(successor)
        all_successors.update(get_all_successors(G, successor, memo))

    memo[node] = all_successors
    return all_successors

def compute_tracks(G, subgraphs, subgraph_nodes=None):
    """
    优化后的 compute_tracks 函数，使用缓存和高效的数据结构来提高性能。
    """
    import time
    import sys
    import networkx as nx

    track_map = {}
    time_track = {}
    vo_size_map = {}

    # 初始化所有节点的 track 值
    for node in G.nodes():
        track_map[node] = hash_data(node)

    if subgraph_nodes is None:
        subgraph_nodes = list(G.nodes())
    subgraph_nodes_set = set(subgraph_nodes)

    # 分离非 _replica 和 _replica 节点
    replicate_nodes = [node for node in subgraph_nodes if "_replica" in str(node)]
    original_nodes = [node for node in subgraph_nodes if "_replica" not in str(node)]
    original_nodes_set = set(original_nodes)

    # 创建子图内节点集合的缓存
    subgraphs_nodes_sets = {index: set(nodes) for index, nodes in subgraphs.items()}

    # 缓存已经计算过的节点
    computed_nodes = set()

    # 使用拓扑排序保证节点的依赖关系
    topo_sort = list(nx.topological_sort(G))

    # 优化：使用缓存避免重复计算
    successors_memo = {}

    def get_successors(node, relevant_nodes):
        if node in successors_memo:
            return successors_memo[node]
        successors = [succ for succ in G.successors(node) if succ in relevant_nodes]
        successors_memo[node] = successors
        return successors

    # 处理非 _replica 节点
    for node in reversed(topo_sort):
        if node not in original_nodes_set:
            continue

        if node in computed_nodes:
            continue  # 已经计算过，跳过

        start_time = time.perf_counter()

        # 找到节点所属的子图
        node_subgraphs = [index for index, nodes_set in subgraphs_nodes_sets.items() if node in nodes_set]

        # 合并所有所属子图的节点集合
        relevant_nodes = set()
        for index in node_subgraphs:
            relevant_nodes.update(subgraphs_nodes_sets[index])

        # 获取节点的后继，限制在相关的子图节点内
        successors = get_successors(node, relevant_nodes)

        # 更新 track_map
        successors_tracks = [track_map[succ] for succ in successors if succ in track_map]
        if successors_tracks:
            track_map[node] = hash_combine(successors_tracks)

        # 计算 VO 大小
        vo_size = sum(sys.getsizeof(track_map[succ]) for succ in successors)
        vo_size_map[node] = vo_size

        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1_000_000
        time_track[node] = elapsed_us

        computed_nodes.add(node)

    # 处理 _replica 节点
    for node in replicate_nodes:
        node_str = str(node)
        original_node = node_str.replace("_replica", "")
        if original_node not in track_map:
            continue  # 原始节点未计算，跳过

        start_time = time.perf_counter()

        # 获取所有后继节点，限制在 subgraph_nodes 内
        all_successors = nx.descendants(G, original_node)
        all_successors = all_successors & subgraph_nodes_set

        # 使用缓存避免重复计算
        successors_tracks = [track_map[succ] for succ in all_successors if succ in track_map]
        if successors_tracks:
            track_map[node] = hash_combine(successors_tracks)
        else:
            track_map[node] = track_map[original_node]

        # 计算 VO 大小
        vo_size = sum(sys.getsizeof(track_map[succ]) for succ in all_successors)
        vo_size_map[node] = vo_size

        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1_000_000
        time_track[node] = elapsed_us

    return track_map, time_track, vo_size_map


def trivial_scheme(G):
    """
    计算每个节点的 trace 和 track，同时返回 VO 的大小。
    VO trace 包括链上所有前驱节点的 hash 值，track 包括链上所有后继节点的 hash 值。
    """
    topo_sort = list(nx.topological_sort(G))
    trace_map = {}
    track_map = {}
    time_trivial_trace = {}
    time_trivial_track = {}
    vo_size_trace_map = {}
    vo_size_track_map = {}

    # 使用缓存（memoization）来存储已经计算过的结果
    precursor_hashes_memo = {}
    successors_hashes_memo = {}

    def get_all_precursor_hashes(node):
        if node in precursor_hashes_memo:
            return precursor_hashes_memo[node]
        all_precursor_hashes = set()
        all_precursor_hashes.add(hash_data(node))
        for precursor in G.predecessors(node):
            all_precursor_hashes.update(get_all_precursor_hashes(precursor))
        precursor_hashes_memo[node] = all_precursor_hashes
        return all_precursor_hashes

    def get_all_successors_hashes(node):
        if node in successors_hashes_memo:
            return successors_hashes_memo[node]
        all_successors_hashes = set()
        all_successors_hashes.add(hash_data(node))
        for successor in G.successors(node):
            all_successors_hashes.update(get_all_successors_hashes(successor))
        successors_hashes_memo[node] = all_successors_hashes
        return all_successors_hashes

    for node in topo_sort:
        # 计算 trace
        start_time = time.perf_counter()
        all_precursor_hashes = get_all_precursor_hashes(node)
        trace_map[node] = hash_combine(all_precursor_hashes)
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1_000_000
        time_trivial_trace[node] = elapsed_us

        # 计算 trace VO 大小
        vo_size_trace = sum(sys.getsizeof(h) for h in all_precursor_hashes)
        vo_size_trace_map[node] = vo_size_trace

        # 计算 track
        start_time = time.perf_counter()
        all_successors_hashes = get_all_successors_hashes(node)
        track_map[node] = hash_combine(all_successors_hashes)
        end_time = time.perf_counter()
        elapsed_us = (end_time - start_time) * 1_000_000
        time_trivial_track[node] = elapsed_us

        # 计算 track VO 大小
        vo_size_track = sum(sys.getsizeof(h) for h in all_successors_hashes)
        vo_size_track_map[node] = vo_size_track

    return trace_map, track_map, time_trivial_trace, time_trivial_track, vo_size_trace_map, vo_size_track_map

def create_subgraphs(G, k):
    subgraphs = {}
    nodes = list(G.nodes())
    for i, node in enumerate(nodes):
        subgraph_index = i // k + 1
        if subgraph_index not in subgraphs:
            subgraphs[subgraph_index] = []
        subgraphs[subgraph_index].append(node)
    return subgraphs

if sys.version_info < (3, 6):
    try:
        import sha3
    except:
        from warnings import warn
        warn("sha3 is not working!")


class MerkleTools(object):
    def __init__(self, hash_type="sha256"):
        hash_type = hash_type.lower()
        if hash_type in ['sha256', 'md5', 'sha224', 'sha384', 'sha512',
                         'sha3_256', 'sha3_224', 'sha3_384', 'sha3_512']:
            self.hash_function = getattr(hashlib, hash_type)
        else:
            raise Exception('`hash_type` {} nor supported'.format(hash_type))

        self.reset_tree()

    def _to_hex(self, x):
        try:  # python3
            return x.hex()
        except:  # python2
            return binascii.hexlify(x)

    def reset_tree(self):
        self.leaves = list()
        self.levels = None
        self.is_ready = False

    def add_leaf(self, values, do_hash=False):
        self.is_ready = False
        # check if single leaf
        if not isinstance(values, tuple) and not isinstance(values, list):
            values = [values]
        values.sort()
        for v in values:
            if do_hash:
                v = v.encode('utf-8')
                v = self.hash_function(v).hexdigest()
            v = bytearray.fromhex(v)
            self.leaves.append(v)

    def insert_leaf(self, value, do_hash=False):
        """
        Inserts a new leaf into the MT, maintaining ascending order.
        """
        self.is_ready = False
        # Hash the value if required
        if do_hash:
            value = self.hash_function(value.encode('utf-8')).digest()
        else:
            value = bytes.fromhex(value)
        # Insert the new leaf into the sorted list of leaves
        self.leaves.append(value)
        self.leaves.sort()
        # Rebuild the tree
        self.make_tree()

    def get_leaf(self, index):
        return self._to_hex(self.leaves[index])

    def get_leaf_count(self):
        return len(self.leaves)

    def get_tree_ready_state(self):
        return self.is_ready

    def _calculate_next_level(self):
        solo_leave = None
        N = len(self.levels[0])  # number of leaves on the level
        if N % 2 == 1:  # if odd number of leaves on the level
            solo_leave = self.levels[0][-1]
            N -= 1

        new_level = []
        for l, r in zip(self.levels[0][0:N:2], self.levels[0][1:N:2]):
            new_level.append(self.hash_function(l+r).digest())
        if solo_leave is not None:
            new_level.append(solo_leave)
        self.levels = [new_level, ] + self.levels  # prepend new level

    def make_tree(self):
        self.is_ready = False
        if self.get_leaf_count() > 0:
            self.levels = [self.leaves, ]
            while len(self.levels[0]) > 1:
                self._calculate_next_level()
        self.is_ready = True

    def get_merkle_root(self):
        if self.is_ready:
            if self.levels is not None:
                return self._to_hex(self.levels[0][0])
            else:
                return None
        else:
            return None


    def get_proof(self, index):
        if self.levels is None:
            return None, 0
        elif not self.is_ready or index > len(self.leaves) - 1 or index < 0:
            return None, 0
        else:
            total_size = 0
            proof = []
            total_size += sys.getsizeof(proof)
            for x in range(len(self.levels) - 1, 0, -1):
                level_len = len(self.levels[x])
                if (index == level_len - 1) and (level_len % 2 == 1):  # skip if this is an odd end node
                    index = int(index / 2.)
                    continue
                is_right_node = index % 2
                sibling_index = index - 1 if is_right_node else index + 1
                sibling_pos = "left" if is_right_node else "right"
                sibling_value = self._to_hex(self.levels[x][sibling_index])

                sibling_size = sys.getsizeof(sibling_value)
                total_size += sibling_size

                proof.append({sibling_pos: sibling_value})
                total_size += sys.getsizeof(proof[-1])
                index = int(index / 2.)
            return proof,total_size

    def validate_proof(self, proof, target_hash, merkle_root):
        merkle_root = bytearray.fromhex(merkle_root)
        target_hash = bytearray.fromhex(target_hash)
        if len(proof) == 0:
            return target_hash == merkle_root
        else:
            proof_hash = target_hash
            for p in proof:
                try:
                    # the sibling is a left node
                    sibling = bytearray.fromhex(p['left'])
                    proof_hash = self.hash_function(sibling + proof_hash).digest()
                except:
                    # the sibling is a right node
                    sibling = bytearray.fromhex(p['right'])
                    proof_hash = self.hash_function(proof_hash + sibling).digest()
            return proof_hash == merkle_root


def create_smpts(G, subgraphs):
    results = {}
    for index, subgraph_nodes in subgraphs.items():
        storage={}
        mpt=MerklePatriciaTrie(storage)
        smt={}
        cross_group_edges = []
        ids=[]
        id_subgraph={}
        mts={}

        # 找到跨组边
        for node in subgraph_nodes:
            for successor in G.successors(node):
                successor_group = [idx for idx, nodes in subgraphs.items() if successor in nodes][0]
                if successor_group != index:
                    edge = (node, successor_group)
                    if edge not in cross_group_edges:
                        cross_group_edges.append((node, successor_group))

        # 如果存在跨组边，则对这些边进行排序并创建 Merkle 树
        if cross_group_edges:
            sorted_edges = sorted(cross_group_edges, key=lambda x: (x[0], x[1]))
            #edges_str_list = [f"{edge[0]}-{edge[1]}" for edge in sorted_edges]
            for edge in sorted_edges:
                if edge[0] not in ids:
                    ids.append(edge[0])
                    id_subgraph[edge[0]]=[]
                id_subgraph[edge[0]].append(edge[1])
            #创建每个id的smt
                for id,values in id_subgraph.items():
                    mt=MerkleTools()

                    for value in values:
                        mt.add_leaf(str(value),True)

                    mt.make_tree()
                    merkle_root=mt.get_merkle_root()
                    smt[id]=merkle_root
                    mts[id]=mt

            #把所有id存进mpt里，但实际上没smt的不需要mpt
                for i in ids:
                    mpt.update(i.encode(),smt[i].encode())

        #print(ids)
        #print(id_subgraph)

        if ids:
            results[index] = {
                'mpt': mpt,
                'smt': smt,
                'mts': mts,
                "link":id_subgraph
            }

    # 返回所有子图的MPT和SMT
    return results


def smpt_prove(smpt_results, subgraph_index, id, element):
    vo_smpt = 0
    if subgraph_index not in smpt_results:
        return False

    mpt = smpt_results[subgraph_index]['mpt']
    smt = smpt_results[subgraph_index]['mts']

    root_hash = mpt.root_hash()

    # Step 1: 获取并验证 MPT 证明
    try:
        proof, vo_mpt = mpt.get_proof(id.encode())
    except KeyError as e:
        # 如果 MPT 中找不到键，则打印错误信息并跳过此项
        print(f"KeyError: {e} - ID: {id} 不在 MPT 中")
        return False

    # 验证 MPT 证明
    is_valid, retrieved_value = MerklePatriciaTrie.verify_proof(root_hash, id.encode(), proof)

    if not is_valid:
        # 如果 MPT 证明无效，打印错误信息并跳过此项
        print(f"ID {id} 不在 MPT 中，验证失败")
        return False

    # Step 2: 验证 SMT 中是否包含 element
    mt = smt[str(id)]

    mt_root = mt.get_merkle_root()

    try:
        # 找到该元素在 SMT 中的位置
        index = mt.leaves.index(mt.hash_function(element.encode('utf-8')).digest())
    except ValueError:
        # 如果找不到该元素，捕获 ValueError 异常并跳过此项
        print(f"Element {element} 在 SMT 中不存在")
        return False

    # 生成 SMT 证明
    smt_proof, vo_smt = mt.get_proof(index)

    # 验证 SMT 证明
    if not mt.validate_proof(smt_proof, mt._to_hex(mt.leaves[index]), mt_root):
        print(f"Element {element} 号位元素在 ID {id} 的 SMT 中的证明验证失败")
        return False

    # 验证通过，返回合并的 VO
    vo_smpt = vo_smt + vo_mpt
    return True, vo_smpt


def find_active_nodes(G, k):
    """
    查找图中所有活跃节点，出度大于 k 的节点。
    """
    active_nodes = []

    for node in G.nodes():
        if G.out_degree(node) > k:
            active_nodes.append(node)

    return active_nodes


def create_single_replica(G, active_nodes):
    """
    为每个活跃节点创建一个复制节点，并将复制节点加入原图中。
    """
    replica_mapping = {}  # 用于存储每个节点的复制节点映射

    for node in active_nodes:
        # 为每个活跃节点创建一个复制节点
        replica_node = f"{node}_replica"

        # 在图中添加复制节点
        G.add_node(replica_node)

        # 将当前节点与复制节点进行映射
        replica_mapping[node] = replica_node

    return replica_mapping


def redirect_edges_to_single_replica(G, active_nodes, replica_mapping):
    """
    将活跃节点的边重定向到其唯一的复制节点。
    """
    for node in active_nodes:
        # 获取对应的复制节点
        replica_node = replica_mapping[node]

        # 重定向当前节点的出边到复制节点
        edges_to_redirect = list(G.out_edges(node))
        G.remove_edges_from(edges_to_redirect)

        # 将所有边都指向复制节点
        for (src, target) in edges_to_redirect:
            G.add_edge(node, replica_node)


def trace_from_active_data(G, node, replica_mapping):
    """
    在追溯过程中，如果遇到活跃节点，直接从其最新复制节点开始追溯。
    """
    # 判断当前节点是否是活跃节点
    if node in replica_mapping:
        # 如果是活跃节点，使用其最新的复制节点
        node = replica_mapping[node]

    # 在这里进行正常的追溯操作，使用最新的复制节点进行追溯
    trace = [node]  # 示例，实际应该根据需要追溯的逻辑进行修改
    while G.in_degree(node) > 0:
        node = list(G.predecessors(node))[0]
        trace.append(node)

    return trace

'''
def update_active_nodes(G, active_nodes, replica_arrays):
    """
    更新活跃节点并重新检查复制节点是否应当被标记为活跃节点。
    """
    new_active_nodes = []
    for idx, node in enumerate(active_nodes):
        for replica in replica_arrays[idx]:
            if G.out_degree(replica) > 3 and replica not in active_nodes:
                active_nodes.append(replica)
                new_active_nodes.append(replica)

                # 为新活跃的复制节点创建新的复制节点
                replica_node = f"{replica}_replica"
                G.add_node(replica_node)
                replica_arrays.append([replica_node])  # 加入新复制节点

    return active_nodes, replica_arrays

'''





def main(filename):
    G = read_graph_from_file(filename)
    G, predecessors, successors = remove_cycles_dfs(G)
    # 连接前驱和后继节点，确保不引入新的环
    G = connect_without_cycles(G, predecessors, successors)
    print(1)
    trivial_trace, trivial_track, time_tri_trace, time_tri_track,vo_triTrace,vo_tritrack = trivial_scheme(G)
    print(2)
    # active_nodes = find_active_nodes(G, 10)
    # replica_mapping = create_single_replica(G, active_nodes)
    # redirect_edges_to_single_replica(G, active_nodes, replica_mapping)

    # 创建复制节点，并更新映射
    # for node in active_nodes:
    #     # 为每个活跃节点创建一个复制节点，并在复制节点映射中记录
    #     if node not in replica_mapping:
    #         replica_mapping[node] = f"{node}_replica"
    #     else:
    #         # 如果节点已有复制节点，则更新为最新复制节点
    #         replica_mapping[node] = f"{node}_replica"

    # 重定向边到复制节点

    subgraphs = create_subgraphs(G, 10)
    trace_results, time_trace,vo_trace = compute_traces(G)
    print(3)
    track_results,time_track,vo_track = compute_tracks(G, subgraphs)
    print(4)
    #print(vo_trace)

    smpt_results = create_smpts(G, subgraphs)
    vo_smpt = {}
    time_set = defaultdict(lambda: 0)

    for index, result in smpt_results.items():
        link = result.get('link', {})
        for key, values in link.items():
            for value in values:
                # 记录开始时间
                start_time = time.perf_counter()

                # 调用 smpt_prove 函数并处理返回值
                result = smpt_prove(smpt_results, index, str(key), str(value))

                if result is False:
                    # 如果验证失败，直接跳过该节点
                    continue

                # 解包返回值（如果验证成功）
                is_valid, vo_smpt[key] = result

                # 记录结束时间
                end_time = time.perf_counter()

                # 计算时间差，转换为微秒
                elapsed_us = (end_time - start_time) * 1_000_000
                time_set[key] += elapsed_us

    # 汇总时间统计
    time_track_total = {}
    for key in time_set:
        time_track_total[key] = time_set[key]

    for key in time_track:
        if key in time_track_total:
            time_track_total[key] += time_track[key]  # 如果key在result中，则进行相加
        else:
            time_track_total[key] = time_track[key]

    # 汇总 VO 数据
    vo_track_total = {}
    for key in vo_smpt:
        vo_track_total[key] = vo_smpt[key]

    for key in vo_track:
        if key in vo_track_total:
            vo_track_total[key] += vo_track[key]  # 如果key在result中，则进行相加
        else:
            vo_track_total[key] = vo_track[key]

    # 将结果转换为字符串类型
    time_trace = {str(k): v for k, v in time_trace.items()}
    time_track_total = {str(k): v for k, v in time_track_total.items()}
    time_tri_trace = {str(k): v for k, v in time_tri_trace.items()}
    time_tri_track = {str(k): v for k, v in time_tri_track.items()}
    vo_trace = {str(k): v for k, v in vo_trace.items()}
    vo_track_total = {str(k): v for k, v in vo_track_total.items()}
    vo_triTrace = {str(k): v for k, v in vo_triTrace.items()}
    vo_tritrack = {str(k): v for k, v in vo_tritrack.items()}


    #print(time_track)

    df = pd.DataFrame({
        'Trace': time_trace,
        'Track(optimized)': time_track_total,
        'tri_Trace': time_tri_trace,
        'tri_track': time_tri_track,
        'VO_trace':vo_trace,
        'vo_track_total':vo_track_total,
        'vo_triTrace':vo_triTrace,
        'vo_tritrack':vo_tritrack
    })
    df.to_excel('output_cite_6400.xlsx', index=True)

    #print(f'avange time:{avange}us')
        #print(f"Subgraph {index} MPT: {result['mpt']}")
        #print(f"Subgraph {index} SMT: {result['smt']}")
        #for id, mt in result['mts'].items():
            #print(f"SMT for ID {id} Merkle Root 1 proof: {mt.get_proof(1)}")

    #smpt_prove(smpt_results, 1, '10', '4')
    #smpts = create_smpts(G, subgraphs)
    #print(trace_results)
    #print(track_results)
    #print(smpts)
    #for index, result in smpt_results.items():
        #for id, mt in result['mts'].items():
            #print(f"SMT for ID {id} Merkle Root 1 proof: {mt.get_proof(1)}")

if __name__ == "__main__":
    main('dataset_cite_6400.txt')