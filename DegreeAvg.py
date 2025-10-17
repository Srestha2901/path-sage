import networkx as nx
import matplotlib.pyplot as plt
import json
import ReadGraph

def compute_static_graph_statistics(G, start_time, end_time):
    verts = G.vertices
    n = len(verts)
    m = float(end_time - start_time + 1)  # include both start and end

    avg_statistics = dict.fromkeys(verts, 0)

    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(verts)

    for t in range(start_time, end_time + 1):
        # add edges from this snapshot to aggregated graph
        aggregated_graph.add_edges_from(G.snapshots[t].edges())

        # compute betweenness for this snapshot
        bc = nx.degree_centrality(G.snapshots[t])
        for v in verts:
            avg_statistics[v] += bc[v]

    # average over all snapshots
    for v in verts:
        avg_statistics[v] /= m
    with open('avg_degree.txt', 'w') as agg_file:
        json.dump(avg_statistics, agg_file, indent=4)

    return avg_statistics

input_file_path = 'file.txt'  # Update this to your actual file path

G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
compute_static_graph_statistics(G2, start_time, end_time)
