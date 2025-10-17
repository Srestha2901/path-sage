import networkx as nx
from scipy.sparse import csr_matrix
import json

def graph_to_adj_bet(graph):
    # Load the graph from the text file
    G = nx.read_weighted_edgelist('file.txt', nodetype=int, create_using=nx.MultiGraph())

    # Calculate and store valid paths for each node with node sequence
    valid_paths_for_nodes = {}
    for node in G.nodes():
        valid_paths = []
        for neighbor1 in G.neighbors(node):
            for neighbor2 in G.neighbors(node):
                if neighbor1 != neighbor2:
                    # Get all edge data between node and neighbor1 and neighbor2
                    edges1 = G.get_edge_data(node, neighbor1)
                    edges2 = G.get_edge_data(node, neighbor2)

                    # Iterate through all edges and their respective weights
                    for key1, data1 in edges1.items():
                        for key2, data2 in edges2.items():
                            weight1 = data1['weight']
                            weight2 = data2['weight']
                            if weight1 < weight2:
                                try:
                                    #path = nx.shortest_path(G, source=neighbor1, target=neighbor2, weight='weight')
                                    path = [neighbor1, node, neighbor2]
                                    valid_paths.append(path)
                                except nx.NetworkXNoPath:
                                    continue
        valid_paths_for_nodes[node] = valid_paths
    # print("valid_paths_for_nodes", valid_paths_for_nodes)

    # Calculate temporal aggregated score for each node
    temporal_aggregated_scores = {str(node): len(valid_paths) for node, valid_paths in valid_paths_for_nodes.items()}

    # Calculate total valid paths sum
    total_valid_paths_sum = sum(temporal_aggregated_scores.values())

    # Calculate temporal average score for each node
    temporal_average_scores = {str(node): score / total_valid_paths_sum for node, score in
                               temporal_aggregated_scores.items()}

    # Print or use the results as needed
    print("Temporal Aggregated Scores:")
    # for node, score in temporal_aggregated_scores.items():
    #     print(f"Node {node}: {score}")

    sorted_agg_statistics_by_node = {int(k): v for k, v in
                                     sorted(temporal_aggregated_scores.items(), key=lambda item: int(item[0]))}
    sorted_nodes = [int(k) for k, v in
                    sorted(temporal_aggregated_scores.items(), key=lambda item: item[1], reverse=True)]
    # print("Sorted Nodes by Score:", sorted_nodes)

    with open('lpc.txt', 'w') as agg_file:
        json.dump(sorted_agg_statistics_by_node, agg_file, indent=4)


    return temporal_aggregated_scores


input_file_path = 'file.txt'
# G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
graph_to_adj_bet(input_file_path)
