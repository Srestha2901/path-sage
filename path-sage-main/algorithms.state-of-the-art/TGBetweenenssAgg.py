import numpy as np
import json

# Step 1: Read nodes from input file
input_file_path = 'file.txt'
nodes_set = set()
with open(input_file_path, 'r') as f:
    for line in f:
        u, v, t = line.strip().split()
        nodes_set.update([int(u), int(v)])

nodes = sorted(list(nodes_set))
# print("Nodes from input (sorted):", nodes)

# Step 2: Load temporal shortest matrix
matrix_file_path = 'temporal_shortest_matrix.txt'
with open(matrix_file_path, 'r') as f:
    data = json.load(f)

temporal_shortest_matrix = np.array(data['temporal_shortest_matrix'])
# print("\nLoaded Temporal Shortest Matrix:")
# print(temporal_shortest_matrix)

# Step 3: Load centrality values from agg_bet.txt
centrality_file = 'agg_bet.txt'
with open(centrality_file, 'r') as f:
    centrality_data = json.load(f)

# Step 4: Function to get distance
def get_distance(u_label, v_label):
    i = nodes.index(u_label)
    j = nodes.index(v_label)
    return temporal_shortest_matrix[i, j]

# Step 5: Set R
R = 3

# Step 6: Compute contributions for all node pairs with distance <= R
# print(f"\nPairwise contributions (distance <= {R}):")
TG = {u: 0 for u in nodes}  # initialize total TG per node

for u in nodes:
    for v in nodes:
        if u != v:
            dist = get_distance(u, v)
            if 0.0001 < dist <= R and not np.isinf(dist):  # distance > 0 and ≤ R
                ci = centrality_data.get(str(u), 0)
                cj = centrality_data.get(str(v), 0)
                contrib = (ci * cj) / (dist ** 2)
                TG[u] += contrib  # sum contribution node-wise
                # print(f"{u} <- {v} | {ci} * {cj} / {dist} = {contrib}")

# Step 7: Print total TG per node
# print("\nTotal Temporal Gravity per node:")
# for node, total in TG.items():
#     print(f"Node {node} -> TG = {total}")

# Step 8: Save TG dictionary to a file
output_file = 'TG_Aggregated_bet.txt'
with open(output_file, 'w') as f:
    json.dump({str(k): float(v) for k, v in TG.items()}, f, indent=4)

