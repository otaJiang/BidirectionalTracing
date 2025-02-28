import networkx as nx
from collections import defaultdict
import random

def filter_citation_network(input_file, output_file, target_nodes):
    G = nx.DiGraph()
    
    # 读取原始数据
    with open(input_file, 'r') as f:
        for line in f:
            dst, src = line.strip().split('\t')
            G.add_edge(src, dst)
    
    node_degrees = {node: G.in_degree(node) + G.out_degree(node) 
                   for node in G.nodes()}
    
    sorted_nodes = sorted(node_degrees.items(), 
                        key=lambda x: x[1], 
                        reverse=True)
    # 优先选择度数大的节点
    selected_nodes = set()
    for node, degree in sorted_nodes:
        if len(selected_nodes) >= target_nodes:
            break
            
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        if len(selected_nodes & neighbors) > 0 or len(selected_nodes) == 0:
            selected_nodes.add(node)
    
    subgraph = G.subgraph(selected_nodes)
    
    # 获取边数
    edge_count = subgraph.number_of_edges()
    print(f"筛选后的子图包含 {edge_count} 条边")
    
    with open(output_file, 'w') as f:
        for edge in subgraph.edges():
            f.write(f"{edge[1]}\t{edge[0]}\n")
    
    return subgraph, edge_count


_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_100.txt', 100)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_200.txt', 200)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_400.txt', 400)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_500.txt', 500)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_800.txt', 800)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_1000.txt', 1000)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_1600.txt', 1600)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_3200.txt', 3200)
_, edge_count = filter_citation_network('Cit-HepPh.txt', 'dataset_cite_6400.txt', 6400)

