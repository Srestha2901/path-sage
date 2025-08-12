import networkx as nx
import json
import ReadGraph
import time
def compute_static_graph_statistics(G, start_time, end_time):
    start_runtime = time.time()
    verts = G.vertices
    print("verts :", verts)
    n = len(verts)
    m = float(end_time - start_time)
    avg_statistics = dict.fromkeys(verts, 0)

    aggregated_graph = nx.Graph()
    aggregated_graph.add_nodes_from(verts)
    start_time = max(1, start_time)
    for t in range(start_time, end_time + 1):
        aggregated_graph.add_edges_from(G.snapshots[t].edges())
        cc = nx.closeness_centrality(G.snapshots[t])
        for v in verts:
            avg_statistics[v] += cc[v]
    nx.draw(aggregated_graph, with_labels=True)
    for v in verts:
        avg_statistics[v] = avg_statistics[v] / m



    print("Avg Statistics:", avg_statistics)

    with open('avg_closeness.txt', 'w') as avg_file:
        json.dump(avg_statistics, avg_file, indent=4)
    end_runtime = time.time()  # End timer
    print(f"Runtime: {end_runtime - start_runtime:.4f} seconds")  # Print runtime

    return  avg_statistics


input_file_path = 'input1.txt'  # Update this to your actual file path
G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)

compute_static_graph_statistics(G2, start_time, end_time)
