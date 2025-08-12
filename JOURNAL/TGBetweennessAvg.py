import numpy as np
import ReadGraph
import json
import time

start_runtime = time.time()
def temporal_gravity_model3(G, avg_centrality, distance_type, R, fastest_arrival_matrix, temporal_shortest_matrix):

    if distance_type == 'fad':
        distance_matrix = fastest_arrival_matrix
    elif distance_type == 'tsd':
        distance_matrix = temporal_shortest_matrix
    else:
        raise ValueError("Invalid distance type. Use 'fad' for fastest arrival distance or 'tsd' for temporal shortest distance.")
    nodes = G.nodes
    TG = {node: 0 for node in nodes}

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if i != j:
                distance = distance_matrix[i, j]
                if distance is not None and distance <= R and distance > 0.0001:
                    TG[u] += (avg_centrality[u]['average_betweenness'] * avg_centrality[v]['average_betweenness']) / (distance ** 2)

    return TG

# def temporal_gravity_model4(G, agg_centrality, distance_type, R, fastest_arrival_matrix, temporal_shortest_matrix):
#     #nodes, fastest_arrival_matrix, temporal_shortest_matrix, _, _ = DistanceMatrix.compute_distance_matrices(G)
#     if distance_type == 'fad':
#         distance_matrix = fastest_arrival_matrix
#     elif distance_type == 'tsd':
#         distance_matrix = temporal_shortest_matrix
#     else:
#         raise ValueError("Invalid distance type. Use 'fad' for fastest arrival distance or 'tsd' for temporal shortest distance.")
#     nodes = G.nodes
#     TG = {node: 0 for node in nodes}
#
#     for i, u in enumerate(nodes):
#         for j, v in enumerate(nodes):
#             if i != j:
#                 distance = distance_matrix[i, j]
#                 if distance is not None and distance <= R and distance > 0.0001:
#                     TG[u] += (agg_centrality[u]['agg_betweenness'] * agg_centrality[v]['agg_betweenness']) / (distance ** 2)
#
#     return TG


def load_matrix_from_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    size = len(data['nodes'])
    matrix = np.zeros((size, size))

    if 'fastest_arrival_matrix' in data:
        for i in range(size):
            for j in range(size):
                matrix[i, j] = data['fastest_arrival_matrix'][i][j]
    elif 'temporal_shortest_matrix' in data:
        for i in range(size):
            for j in range(size):
                matrix[i, j] = data['temporal_shortest_matrix'][i][j]
    else:
        raise ValueError("Matrix type ('fastest_arrival_matrix' or 'temporal_shortest_matrix') not found in JSON.")

    return matrix

# Example usage:
file_path_fastest = 'fastest_arrival_matrix.txt'
fastest_arrival_matrix = load_matrix_from_json(file_path_fastest)

file_path_temporal = 'temporal_shortest_matrix.txt'
temporal_shortest_matrix = load_matrix_from_json(file_path_temporal)

print("Fastest Arrival Matrix:")
print(fastest_arrival_matrix)
print("\nTemporal Shortest Matrix:")
print(temporal_shortest_matrix)

input_file_path = 'input1.txt'

G1, G2, end_time, start_time, edges = ReadGraph.read_graph_from_file(input_file_path)
R = 3

distance_type = 'fad'
#nodes, fastest_arrival_matrix, temporal_shortest_matrix, fastest_arrival_paths, temporal_shortest_paths = DistanceMatrix.compute_distance_matrices(G1)

with open('avg_bet.txt') as f:
    data = f.read()
avg_statistics = json.loads(data)

centrality_matrix = {}
verts = list(G2.vertices)

for v in verts:
    centrality_matrix[v] = {
        'average_betweenness': avg_statistics.get(v, 0)
    }

distance_type = 'tsd'

# avg betweenness Centrality is used as a Mass
TG = temporal_gravity_model3(G1, centrality_matrix, distance_type, R, fastest_arrival_matrix, temporal_shortest_matrix)

# Convert np.float64 values to standard Python floats
temporal_gravity_dict_betweenness_tsd = {str(node): float(tg_value) for node, tg_value in TG.items()}
print("Temporal Gravity on betweenness_tsd:", temporal_gravity_dict_betweenness_tsd)
end_runtime = time.time()  # End timer
print(f"Runtime: {end_runtime - start_runtime:.4f} seconds")  # Print runtime

with open('TG_average_betweenness_tsd.txt', 'w') as file:
    file.write(json.dumps(temporal_gravity_dict_betweenness_tsd))