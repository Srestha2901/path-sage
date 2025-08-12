import ReadGraph
import json
import numpy as np
import time

start_runtime = time.time()
def temporal_gravity_model4(G, agg_centrality, distance_type, R, fastest_arrival_matrix, temporal_shortest_matrix):
    # Determine which distance matrix to use
    if distance_type == 'fad':
        distance_matrix = fastest_arrival_matrix
    elif distance_type == 'tsd':
        distance_matrix = temporal_shortest_matrix
    else:
        raise ValueError(
            "Invalid distance type. Use 'fad' for fastest arrival distance or 'tsd' for temporal shortest distance.")

    nodes = list(G.nodes())
    node_count = len(nodes)
    TG = {node: 0 for node in nodes}

    agg_degrees = np.array([agg_centrality[node]['agg_degree'] for node in nodes])

    for i in range(node_count):
        for j in range(node_count):
            if i != j:
                distance = distance_matrix[i, j]
                if distance is not None and distance <= R and distance > 0.0001:
                    TG[nodes[i]] += (agg_degrees[i] * agg_degrees[j]) / (distance ** 2)

    return TG
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


with open('agg_degree.txt') as f:
    data = f.read()
agg_statistics = json.loads(data)

centrality_matrix = {}
verts = list(G2.vertices)
for v in verts:
    centrality_matrix[v] = {
        'agg_degree': agg_statistics.get(v, 0),
    }

distance_type = 'tsd'

# agg degree Centrality is used as a Mass
TG = temporal_gravity_model4(G1, centrality_matrix, distance_type, R, fastest_arrival_matrix, temporal_shortest_matrix)

# Convert np.float64 values to standard Python floats
temporal_gravity_dict_degree_agg_tsd = {str(node): float(tg_value) for node, tg_value in TG.items()}
print("Temporal Gravity on degree_agg_tsd:", temporal_gravity_dict_degree_agg_tsd)

end_runtime = time.time()  # End timer
print(f"Runtime: {end_runtime - start_runtime:.4f} seconds")  # Print runtime

with open('TG_Aggregated_degree_tsd.txt', 'w') as file:
    file.write(json.dumps(temporal_gravity_dict_degree_agg_tsd))
