import networkx as nx
import ReadGraph
import json
import time

def graph_to_adj_bet(graph):
    start_runtime = time.time()
    # Load the graph from the text file
    G = nx.read_weighted_edgelist('input1.txt', nodetype=int, create_using=nx.MultiGraph())

    valid_paths_for_nodes = {}

    for node in G.nodes():
        valid_paths = []
        seen_first_middle = set()  # track only first and middle nodes

        for neighbor1 in G.neighbors(node):
            for neighbor2 in G.neighbors(node):
                if neighbor1 != neighbor2:
                    edges1 = G.get_edge_data(node, neighbor1)
                    edges2 = G.get_edge_data(node, neighbor2)

                    for _, data1 in edges1.items():
                        for _, data2 in edges2.items():
                            weight1 = data1['weight']
                            weight2 = data2['weight']
                            if weight1 < weight2:
                                first_middle_pair = tuple(sorted((neighbor1, node)))
                                if first_middle_pair not in seen_first_middle:
                                    seen_first_middle.add(first_middle_pair)
                                    path = [neighbor1, node, neighbor2]
                                    valid_paths.append(path)

        valid_paths_for_nodes[node] = valid_paths

    # Print all valid paths
    print("\nValid paths for each node:")
    for node, paths in valid_paths_for_nodes.items():
        print(f"Node {node}:")
        for p in paths:
            print("   ", p)

    # Calculate temporal aggregated score
    temporal_aggregated_scores = {
        str(node): len(valid_paths)
        for node, valid_paths in valid_paths_for_nodes.items()
    }

    print("\nTemporal Aggregated Scores:")
    for node, score in temporal_aggregated_scores.items():
        print(f"Node {node}: {score}")

    with open('lpc.txt', 'w') as agg_file:
        json.dump(temporal_aggregated_scores, agg_file, indent=4)
    end_runtime = time.time()  # End timer
    print(f"Runtime: {end_runtime - start_runtime:.4f} seconds")  # Print runtime

    return temporal_aggregated_scores

# Run
input_file_path = 'input1.txt'
G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
graph_to_adj_bet(G2)
