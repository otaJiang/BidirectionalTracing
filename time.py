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
    """
    从文件读取图，初始化每个节点的时间戳属性为 None。
    """
    G = nx.DiGraph()
    with open(filename, 'r') as file:
        for line in file:
            dst, src = line.strip().split('\t')
            G.add_edge(src, dst)

            # 初始化节点的时间戳属性
            if src not in G.nodes:
                G.nodes[src]['timestamp'] = None
            if dst not in G.nodes:
                G.nodes[dst]['timestamp'] = None

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


def compute_full_traces(G, nodes=None):
    """
    计算图中所有节点的 trace 值。
    :param G: 图对象
    :param nodes: 可选的节点列表，指定要计算 trace 的节点
    :return: trace_map, time_trace, vo_size_map
    """
    if nodes is None:
        nodes = list(G.nodes())
    topo_sort = list(nx.topological_sort(G))
    trace_map = {}
    time_trace = {}
    vo_size_map = {}

    for node in topo_sort:
        if node not in nodes:
            continue

        # 如果是复制节点，找到原本节点
        start_perf = time.perf_counter()
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

        end_perf = time.perf_counter()
        elapsed_us = (end_perf - start_perf) * 1_000_000
        time_trace[node] = elapsed_us

        # 计算 VO 的总大小
        vo_size = sum(sys.getsizeof(el) for el in vo_elements)
        vo_size_map[node] = vo_size

    return trace_map, time_trace, vo_size_map

def compute_trace_for_node_with_time_range(G, node, start_time, end_time):
    """
    计算单个节点的 trace，过滤时间范围内的前驱节点。
    :param G: 图对象
    :param node: 待追溯的节点
    :param start_time: 时间区间起始
    :param end_time: 时间区间结束
    :return: 过滤后的前驱节点列表
    """
    if node not in G.nodes:
        print(f"Warning: Node {node} is not in the graph!")
        return {}

    trace_map = {}  # 存储有效的前驱节点

    # 定义递归函数来追溯节点的前驱
    def trace_node(node, visited):
        if node in visited:
            return []
        visited.add(node)
        predecessors = list(G.predecessors(node))
        valid_predecessors = []

        for precursor in predecessors:
            node_timestamp = G.nodes[precursor].get('timestamp', None)
            if node_timestamp is None:
                continue
            if start_time <= node_timestamp <= end_time:
                valid_predecessors.append(precursor)
                print(f"Node {node} -> Precursor {precursor} within time range [{start_time}, {end_time}]")
                valid_predecessors.extend(trace_node(precursor, visited))
        return valid_predecessors

    # 开始追溯
    valid_predecessors = trace_node(node, set())
    trace_map[node] = list(set(valid_predecessors))  # 去重

    return trace_map

def compute_traces_combined(G, node_to_trace=None, start_time=None, end_time=None, k=10):
    """
    计算全图的 trace，并在需要时计算单个节点的 trace 结果，过滤时间范围内的前驱节点。

    :param G: 图对象
    :param node_to_trace: 待追溯的节点ID（可选）
    :param start_time: 时间区间起始（可选）
    :param end_time: 时间区间结束（可选）
    :param k: 子图大小（默认10）
    :return: 全图 trace_map, time_trace, vo_size_map, 单个节点 trace_results（如果指定）
    """

    # Step 1: 创建子图并分配时间戳
    subgraphs = create_subgraphs_and_assign_timestamps(G, k)
    print("Subgraphs created and timestamps assigned.")

    # Step 2: 打印节点及其时间戳（可选）
    print("Node Timestamps:")
    for node in G.nodes():
        print(f"Node {node}: Timestamp = {G.nodes[node]['timestamp']}")

    # Step 3: 计算全图的 trace
    trace_map, time_trace, vo_size_map = compute_full_traces(G)
    print("Full trace computed.")

    # Step 4: 如果指定了节点和时间区间，计算单个节点的 trace 并过滤时间范围内的前驱节点
    trace_results = {}
    if node_to_trace is not None and start_time is not None and end_time is not None:
        trace_results = compute_trace_for_node_with_time_range(G, node_to_trace, start_time, end_time)
        print(f"Trace results for node {node_to_trace} within time range [{start_time}, {end_time}]:")
        print(trace_results)

    return trace_map, time_trace, vo_size_map, trace_results



def compute_tracks(G, subgraphs, subgraph_nodes=None, node_to_track=None, start_time=None, end_time=None):
    """
    计算全图的 track，并在需要时计算单个节点的 track 结果，过滤时间范围内的前驱节点。

    :param G: 图对象
    :param subgraphs: 子图字典，格式为 {subgraph_index: {'nodes': [node1, node2, ...]}, ...}
    :param subgraph_nodes: 可选，指定要计算 track 的节点列表
    :param node_to_track: 可选，指定要追溯的节点ID
    :param start_time: 可选，时间区间起始
    :param end_time: 可选，时间区间结束
    :return: track_map, time_track, vo_size_map, trace_results（如果指定）
    """

    track_map = {}
    time_track = {}
    vo_size_map = {}
    trace_results = {}

    computed_nodes = set()  # 缓存已计算的节点
    vo_cache = {}  # VO 元素缓存，用于避免重复计算

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
    # 假设 subgraphs 的格式为 {index: {'nodes': [node1, node2, ...]}, ...}
    subgraphs_nodes_sets = {index: set(data['nodes']) for index, data in subgraphs.items()}

    # 使用拓扑排序保证节点的依赖关系
    try:
        topo_sort = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Error: The graph contains a cycle. Topological sort is not possible.")
        return {}, {}, {}, {}

    # Helper function to get node timestamp
    def get_node_timestamp(node):
        return G.nodes[node].get('timestamp', None)

    # Function to perform tracking with time range
    def track_node_with_time_range(node_to_track, start_time, end_time):
        if node_to_track not in G.nodes:
            print(f"Warning: Node '{node_to_track}' is not in the graph!")
            return {}

        valid_predecessors = set()

        def traverse(node, visited):
            if node in visited:
                return
            visited.add(node)
            for precursor in G.predecessors(node):
                node_timestamp = get_node_timestamp(precursor)
                if node_timestamp is None:
                    print(f"  Node '{precursor}' has no timestamp. Skipping.")
                    continue
                if start_time <= node_timestamp <= end_time:
                    if precursor not in valid_predecessors:
                        valid_predecessors.add(precursor)
                        print(f"  Node '{node}' -> Precursor '{precursor}' within time range [{start_time}, {end_time}]")
                        traverse(precursor, visited)

        traverse(node_to_track, set())

        return {node_to_track: list(valid_predecessors)}

    # 处理非 _replica 节点
    for node in reversed(topo_sort):
        if node not in original_nodes_set:
            continue

        if node in computed_nodes:
            continue  # 跳过已计算的节点

        start_perf = time.perf_counter()

        # 获取节点的时间戳
        node_timestamp = get_node_timestamp(node)
        if node_timestamp is None:
            print(f"Error: Node '{node}' does not have a timestamp.")
            continue

        # 找到节点所属的子图
        node_subgraphs = [index for index, nodes_set in subgraphs_nodes_sets.items() if node in nodes_set]

        if not node_subgraphs:
            print(f"Warning: Node '{node}' is not found in any subgraph.")
            continue

        # 合并所有所属子图的节点集合
        relevant_nodes = set()
        for index in node_subgraphs:
            relevant_nodes.update(subgraphs_nodes_sets[index])

        # 获取节点的后继，限制在相关的子图节点内
        successors = [succ for succ in G.successors(node) if succ in relevant_nodes]

        # 更新 track_map
        successors_tracks = [track_map[succ] for succ in successors if succ in track_map]
        if successors_tracks:
            track_map[node] = hash_combine(successors_tracks)

        # 计算 VO 大小，优化 VO 元素的缓存
        vo_elements = []
        for succ in successors:
            if succ in track_map:
                vo_elements.append(track_map[succ])

        vo_size = 0
        for el in vo_elements:
            if el in vo_cache:
                vo_size += vo_cache[el]  # 使用缓存
            else:
                element_size = sys.getsizeof(el)
                vo_cache[el] = element_size  # 缓存计算的元素大小
                vo_size += element_size

        vo_size_map[node] = vo_size

        end_perf = time.perf_counter()
        elapsed_us = (end_perf - start_perf) * 1_000_000  # 微秒
        time_track[node] = elapsed_us

        computed_nodes.add(node)

        print(f"Processed node '{node}': Trace={track_map[node]}, VO Size={vo_size_map[node]}, Elapsed Time={elapsed_us:.2f} us")

    # 处理 _replica 节点
    for node in replicate_nodes:
        node_str = str(node)
        original_node = node_str.replace("_replica", "")
        if original_node not in track_map:
            print(f"Warning: Original node '{original_node}' for replica '{node}' not found in track_map. Skipping.")
            continue  # 原始节点未计算，跳过

        start_perf = time.perf_counter()

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
        vo_size_map[node] = sys.getsizeof(track_map[node])

        end_perf = time.perf_counter()
        elapsed_us = (end_perf - start_perf) * 1_000_000  # 微秒
        time_track[node] = elapsed_us

        print(f"Processed replica node '{node}': Trace={track_map[node]}, VO Size={vo_size_map[node]}, Elapsed Time={elapsed_us:.2f} us")

    # 如果指定了节点和时间区间，计算单个节点的 track 并过滤时间范围内的前驱节点
    if node_to_track and start_time is not None and end_time is not None:
        print(f"\nStarting tracking for node '{node_to_track}' within time range [{start_time}, {end_time}]...")
        trace_results = track_node_with_time_range(node_to_track, start_time, end_time)
    else:
        trace_results = {}

    return track_map, time_track, vo_size_map, trace_results



def create_subgraphs_and_assign_timestamps(G, k):
    """
    根据每个子图的大小 k 来创建子图，并为每个节点分配等差时间戳。
    返回的字典中，key 是子图的索引，value 是一个包含节点的列表。
    同时为图中的每个节点设置 timestamp 属性。
    """
    subgraphs = {}
    nodes = list(G.nodes())
    time_stamp = 1  # 从时间戳1开始

    # 遍历所有节点，按 k 划分子图
    for i, node in enumerate(nodes):
        subgraph_index = i // k + 1  # 计算当前节点属于哪个子图

        # 如果子图索引不存在，初始化子图结构
        if subgraph_index not in subgraphs:
            subgraphs[subgraph_index] = {'nodes': []}

        # 将节点加入子图
        subgraphs[subgraph_index]['nodes'].append(node)

        # 为当前节点设置时间戳
        G.nodes[node]['timestamp'] = time_stamp  # 设置时间戳属性

        # 时间戳递增
        time_stamp += 1

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
        print(f"KeyError: {e} - ID: {id} 不在 MPT 中")
        return False

    # 验证 MPT 证明
    is_valid, retrieved_value = MerklePatriciaTrie.verify_proof(root_hash, id.encode(), proof)

    if not is_valid:
        print(f"ID {id} 不在 MPT 中，验证失败")
        return False

    # Step 2: 验证 SMT 中是否包含 element
    mt = smt[str(id)]
    mt_root = mt.get_merkle_root()

    try:
        index = mt.leaves.index(mt.hash_function(element.encode('utf-8')).digest())
    except ValueError:
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


def save_to_excel(filename, data_dict):
    # 将字典转换为 DataFrame
    df = pd.DataFrame(list(data_dict.items()), columns=['Dataset', 'Value'])
    # 保存为 Excel 文件
    df.to_excel(filename, index=False)

def main(filename):
    G = read_graph_from_file(filename)
    G, predecessors, successors = remove_cycles_dfs(G)
    # 连接前驱和后继节点，确保不引入新的环
    G = connect_without_cycles(G, predecessors, successors)
    print(1)

    k = 10
    subgraphs = create_subgraphs_and_assign_timestamps(G, 10)
    #trace_map, time_trace, vo_size_map = compute_full_traces(G)

    node_to_track = '107329'  # 替换为你的节点ID
    start_time_range = 0
    end_time_range = 400

    # 计算单个节点的 trace 并过滤时间范围
    trace_map, time_trace, vo_size_map, trace_results = compute_tracks(
        G,
        subgraphs=subgraphs,
        subgraph_nodes=None,  # 或者指定要计算 track 的节点列表
        node_to_track=node_to_track,
        start_time=start_time_range,
        end_time=end_time_range
    )

    print(trace_results)
    #
    # print(3)
    # track_results, time_track, vo_track = compute_tracks(G, subgraphs)
    #
    # smpt_results = create_smpts(G, subgraphs)
    # vo_smpt = {}
    # time_set = defaultdict(lambda: 0)
    #
    # for index, result in smpt_results.items():
    #     link = result.get('link', {})
    #     for key, values in link.items():
    #         for value in values:
    #             # 记录开始时间
    #             start_time = time.perf_counter()
    #
    #             # 调用 smpt_prove 函数并处理返回值
    #             result = smpt_prove(smpt_results, index, str(key), str(value))
    #
    #             if result is False:
    #                 # 如果验证失败，直接跳过该节点
    #                 continue
    #
    #             # 解包返回值（如果验证成功）
    #             is_valid, vo_smpt[key] = result
    #
    #             # 记录结束时间
    #             end_time = time.perf_counter()
    #
    #             # 计算时间差，转换为微秒
    #             elapsed_us = (end_time - start_time) * 1_000_000
    #             time_set[key] += elapsed_us
    #
    # # 汇总时间统计
    # time_track_total = {}
    # for key in time_set:
    #     time_track_total[key] = time_set[key]
    #
    # for key in time_track:
    #     if key in time_track_total:
    #         time_track_total[key] += time_track[key]  # 如果key在result中，则进行相加
    #     else:
    #         time_track_total[key] = time_track[key]
    #
    # # 汇总 VO 数据
    # vo_track_total = {}
    # for key in vo_smpt:
    #     vo_track_total[key] = vo_smpt[key]
    #
    # for key in vo_track:
    #     if key in vo_track_total:
    #         vo_track_total[key] += vo_track[key]  # 如果key在result中，则进行相加
    #     else:
    #         vo_track_total[key] = vo_track[key]
    #
    # # 将结果转换为字符串类型
    # time_trace = {str(k): v for k, v in time_trace.items()}
    # time_track_total = {str(k): v for k, v in time_track_total.items()}
    # vo_trace = {str(k): v for k, v in vo_trace.items()}
    # vo_track_total = {str(k): v for k, v in vo_track_total.items()}
    #
    # # print(time_track)
    #
    # df = pd.DataFrame({
    #     'Trace': time_trace,
    #     'Track(optimized)': time_track_total,
    #     'VO_trace': vo_trace,
    #     'vo_track_total': vo_track_total
    # })
    # df.to_excel('output_multi_cite_800.xlsx', index=True)

if __name__ == "__main__":
    main('dataset_cite_800.txt')