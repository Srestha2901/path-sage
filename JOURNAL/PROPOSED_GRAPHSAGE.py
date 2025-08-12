import torch_geometric
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import networkx as nx

class TemporalGraph:
    def __init__(self, time_steps):
        self.t_end = time_steps
        self.time_steps = time_steps
        self.snapshots = [nx.MultiGraph() for _ in range(time_steps + 1)]
        self.vertices = set()

    def add_vertices(self, verts):
        for vert in verts:
            self.vertices.add(vert)
            for snapshot in self.snapshots:
                snapshot.add_node(vert)

    def add_temporal_edges(self, edges):
        for (source, target), (start_time, end_time) in edges:
            for time in range(start_time, end_time + 1):
                if time < len(self.snapshots):
                    self.snapshots[time].add_edge(source, target)

    # Method to return all unique nodes from all snapshots
    def nodes(self):
        unique_nodes = set()
        for snapshot in self.snapshots:
            unique_nodes.update(snapshot.nodes())
        return unique_nodes

    # Optionally, if you want nodes as a list instead of a set
    def get_nodes(self):
        return list(self.nodes())

def read_graph_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    G1 = nx.MultiGraph()

    vertices_list = []
    Tedges = []
    edges = []
    max_time = 0
    min_time = float('inf')  # Initialize to infinity to find the actual minimum time

    for line in lines:
        parts = line.strip().split()
        source, target, time = parts[0], parts[1], int(parts[2])
        if time > max_time:
            max_time = time
        if time < min_time:
            min_time = time
        G1.add_edge(source, target, time=time)
        Tedges.append(((source, target), (time, time)))
        edges.append((source, target, time))
        if source not in vertices_list:
            vertices_list.append(source)
        if target not in vertices_list:
            vertices_list.append(target)

    G2 = TemporalGraph(max_time + 1)
    G2.add_vertices(vertices_list)
    G2.add_temporal_edges(Tedges)

    return G1, G2, max_time, min_time, edges


def graph_to_adj_bet(graph):
    # Load the graph from the text file
    G = nx.read_weighted_edgelist('temporal_data.txt', nodetype=int, create_using=nx.MultiGraph())

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
    print("valid_paths_for_nodes", valid_paths_for_nodes)

    # Calculate temporal aggregated score for each node
    temporal_aggregated_scores = {str(node): len(valid_paths) for node, valid_paths in valid_paths_for_nodes.items()}

    print("Temporal Aggregated Scores:")
    for node, score in temporal_aggregated_scores.items():
        print(f"Node {node}: {score}")


    # Store the results in two different dictionaries
    temporal_aggregated_scores_dict = temporal_aggregated_scores


    # Export to separate text files
    with open('agg_proposed.txt', 'w') as agg_file:
        json.dump(temporal_aggregated_scores_dict, agg_file, indent=4)



    return temporal_aggregated_scores_dict

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(in_channels, out_channels, aggr='mean')  # Aggregation mode: 'mean'
        self.conv2 = torch_geometric.nn.SAGEConv(out_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        # First convolution
        x = self.conv1(x, edge_index)
        print("After conv1 (includes self and neighbor aggregation):\n", x)

        # Activation
        x = F.relu(x)

        # Second convolution
        x = self.conv2(x, edge_index)
        print("After conv2:\n", x)
        return x

# Load the graph data
input_file_path = 'temporal_data.txt'
G1, G2, end_time, start_time, edges = read_graph_from_file(input_file_path)

# Get aggregated scores as features for each node
temporal_aggregated_scores_dict= graph_to_adj_bet(G2)


sorted_nodes = sorted(G2.get_nodes())

node_features = []
for node in sorted_nodes:
    feature = temporal_aggregated_scores_dict.get(str(node), 0)  # default to 0 if not found
    node_features.append([feature])  # Store as a list to match the expected shape

print("Sorted node features:", node_features)

# Convert the node features to a torch tensor
node_features_tensor = torch.tensor(node_features, dtype=torch.float)
print("node_features_tensor", node_features_tensor)

# Ensure edges are bidirectional for an undirected graph
edges_list = []
for snapshot in G2.snapshots:
    edges_list.extend(snapshot.edges())

# Ensure bidirectionality for undirected edges
edges_list_bidirectional = []
for u, v in edges_list:
    edges_list_bidirectional.append((u, v))
    edges_list_bidirectional.append((v, u))  # Add reverse direction for undirected behavior

# Remove duplicate edges
unique_edges_set = set(edges_list_bidirectional)  # Automatically handles duplicates
edges_list_unique = list(unique_edges_set)

# Create node mapping and edge_index
node_mapping = {node: idx for idx, node in enumerate(sorted_nodes)}
edges_list_int = [[node_mapping[u], node_mapping[v]] for u, v in edges_list_unique]
edge_index = torch.tensor(edges_list_int, dtype=torch.long).t().contiguous()

# Create the Data object
data = Data(x=node_features_tensor, edge_index=edge_index)

# Print data to verify
print("Node Features in Data object:")
print(data.x)

print("\nEdge Index (Bidirectional):")
print(data.edge_index)

# Define and use the model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = torch_geometric.nn.SAGEConv(in_channels, out_channels, aggr='mean')  # Aggregation mode: 'mean'
        self.conv2 = torch_geometric.nn.SAGEConv(out_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        # First convolution
        x = self.conv1(x, edge_index)
        print("After conv1 (includes self and neighbor aggregation):\n", x)

        # Activation
        x = F.relu(x)

        # Second convolution
        x = self.conv2(x, edge_index)
        print("After conv2:\n", x)
        return x

# Instantiate and apply the model
model = GraphSAGE(in_channels=1, out_channels=1)
output = model(data.x, data.edge_index)
print("Model output:")
print(output)
