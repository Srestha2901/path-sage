import ReadGraph
import networkx as nx
import json
def compute_static_graph_statistics(G, start_time, end_time):

    verts = G.vertices
    print("verts :", verts)

    agg_statistics = dict.fromkeys(verts, 0)


    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(verts)
    start_time = max(1, start_time)
    for t in range(start_time, end_time + 1):
        aggregated_graph.add_edges_from(G.snapshots[t].edges())
    bc = nx.betweenness_centrality(aggregated_graph)
    for v in verts:
        agg_statistics[v] = bc[v]

    print("Agg Statistics:", agg_statistics)

    # Export to separate text files
    with open('agg_bet.txt', 'w') as agg_file:
        json.dump(agg_statistics, agg_file, indent=4)

    return agg_statistics


input_file_path = 'file.txt'

G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
compute_static_graph_statistics(G2, start_time, end_time)
