import networkx as nx
import matplotlib.pyplot as plt
import json
import ReadGraph

def compute_static_graph_statistics(G, start_time, end_time):

    verts = G.vertices
    print("verts :", verts)
    n = len(verts)
    m = float(end_time - start_time)

    agg_statistics = dict.fromkeys(verts, 0)

    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(verts)
    start_time = max(1, start_time)
    for t in range(start_time, end_time + 1):
        aggregated_graph.add_edges_from(G.snapshots[t].edges())

        dc = G.snapshots[t].degree()

    nx.draw(aggregated_graph, with_labels=True)

    dc = nx.degree_centrality(aggregated_graph)
    for v in verts:
        agg_statistics[v] = dc[v]

    print("Agg Statistics:", agg_statistics)
    # Export to separate text files
    with open('agg_degree.txt', 'w') as agg_file:
        json.dump(agg_statistics, agg_file, indent=4)

    return agg_statistics



input_file_path = 'file.txt'  # Update this to your actual file path

G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
compute_static_graph_statistics(G2, start_time, end_time)

