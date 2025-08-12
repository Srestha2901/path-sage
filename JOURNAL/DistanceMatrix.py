import json
import ReadGraph
import networkx as nx
import numpy as np
import time

start_time = time.time()

def get_valid_paths(graph, paths):
    valid_paths = []
    for path in paths:
        valid = True
        for u, v, w in zip(path[:-2], path[1:-1], path[2:]):
            try:
                edge_uv_times = [data['time'] if isinstance(data, dict) else data for key, data in graph[u][v].items()]
                edge_vw_times = [data['time'] if isinstance(data, dict) else data for key, data in graph[v][w].items()]
            except KeyError:
                valid = False
                break

            if min(edge_uv_times) >= min(edge_vw_times):
                valid = False
                break
        if valid:
            valid_paths.append(path)
    return valid_paths

def get_path_time(graph, path):
    try:
        return max(min(graph[u][v][key]['time'] if isinstance(graph[u][v][key], dict) else graph[u][v][key] for key in graph[u][v])
                   for u, v in zip(path[:-1], path[1:]))
    except KeyError:
        return float('inf')

def fastest_arrival_distance(graph, valid_paths):
    if not valid_paths:
        return None, float('inf')

    fastest_path = min(valid_paths, key=lambda path: get_path_time(graph, path))
    fastest_time = get_path_time(graph, fastest_path)
    return fastest_path, fastest_time

def temporal_shortest_distance(valid_paths):
    if not valid_paths:
        return None, float('inf')

    shortest_path_length = min(len(path) - 1 for path in valid_paths)
    shortest_paths = [path for path in valid_paths if len(path) - 1 == shortest_path_length]
    return shortest_paths, shortest_path_length

def compute_distance_matrices(graph, max_edges=10):
    nodes = list(graph.nodes)
    n = len(nodes)

    fastest_arrival_matrix = np.full((n, n), float('inf'))
    temporal_shortest_matrix = np.full((n, n), float('inf'))
    fastest_arrival_paths = {}
    temporal_shortest_paths = {}

    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if source == target:
                fastest_arrival_matrix[i, j] = 0
                temporal_shortest_matrix[i, j] = 0
                fastest_arrival_paths[(source, target)] = [source]
                temporal_shortest_paths[(source, target)] = [source]
                continue

            paths = list(nx.all_simple_paths(graph, source=source, target=target, cutoff=max_edges))
            if not paths:
                continue

            valid_paths = get_valid_paths(graph, paths)

            f_path, f_distance = fastest_arrival_distance(graph, valid_paths)
            t_paths, t_distance = temporal_shortest_distance(valid_paths)

            fastest_arrival_matrix[i, j] = f_distance
            temporal_shortest_matrix[i, j] = t_distance

            fastest_arrival_paths[(source, target)] = f_path
            temporal_shortest_paths[(source, target)] = t_paths

    return nodes, fastest_arrival_matrix, temporal_shortest_matrix, fastest_arrival_paths, temporal_shortest_paths
# Example usage
input_file_path = 'input1.txt'
G1, G2, max_time, min_time, edges = ReadGraph.read_graph_from_file(input_file_path)

# Compute the matrices for G1 (static graph)
nodes, fastest_arrival_matrix, temporal_shortest_matrix, fastest_arrival_paths, temporal_shortest_paths = compute_distance_matrices(G1, max_edges=3)

# Store results in a dictionary
fastest_arrival_matrix_dict = {
    "nodes": nodes,
    "fastest_arrival_matrix": fastest_arrival_matrix.tolist()
}
temporal_shortest_matrix_dict = {
    "nodes": nodes,
    "temporal_shortest_matrix": temporal_shortest_matrix.tolist()
}


# Save the results to a text file
output_file_path = 'fastest_arrival_matrix.txt'
with open(output_file_path, 'w') as f:
    json.dump(fastest_arrival_matrix_dict, f, indent=4)

output_file_path = 'temporal_shortest_matrix.txt'
with open(output_file_path, 'w') as f:
    json.dump(temporal_shortest_matrix_dict, f, indent=4)


# Print the result in matrix form
print("Fastest Arrival Matrix:")
print(fastest_arrival_matrix)

print("Temporal Shortest Matrix:")
print(temporal_shortest_matrix)

print("Fastest Arrival Paths:")
print(fastest_arrival_paths)

print("Temporal Shortest Paths:")
print(temporal_shortest_paths)

end_time = time.time()

run_time = end_time-start_time
print("Time to run DistanceMatrix : ", run_time)