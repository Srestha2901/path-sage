# graphsage_rankings_unsupervised.py
import ReadGraph
import networkx as nx
import json
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

def graph_to_adj_bet_return_scores(graph_file):
    t0 = time.time()
    G = nx.read_weighted_edgelist('file.txt', nodetype=int, create_using=nx.MultiGraph())
    print("Total number of nodes:", G.number_of_nodes())
    print("Total number of edges:", G.number_of_edges())
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
                                    # path = nx.shortest_path(G, source=neighbor1, target=neighbor2, weight='weight')
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

    temporal_aggregated_scores = {node: len(paths) for node, paths in valid_paths_for_nodes.items()}

    # Save raw scores
    with open('Pathsage1.txt', 'w') as f:
        json.dump({str(k): int(v) for k, v in temporal_aggregated_scores.items()}, f, indent=4)

    sorted_nodes = sorted(temporal_aggregated_scores.keys())
    # print("sorted_nodes",sorted_nodes)
    features_tensor = torch.tensor([[temporal_aggregated_scores[n]] for n in sorted_nodes], dtype=torch.float)

    feat_runtime = time.time() - t0
    print(f"[+] Feature extraction runtime: {feat_runtime:.4f} s  (num_nodes={len(sorted_nodes)})")

    return sorted_nodes, features_tensor, G, feat_runtime

# ---------- GraphSAGE Model ----------
class GraphSAGE3(torch.nn.Module):
    def __init__(self, in_channels, hidden1=64, hidden2=32, out_channels=1):
        super().__init__()
        self.sage1 = SAGEConv(in_channels, hidden1)
        self.sage2 = SAGEConv(hidden1, hidden2)
        self.sage3 = SAGEConv(hidden2, out_channels)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        x = self.sage3(x, edge_index)
        return x

# ---------- Convert NetworkX to edge_index ----------
def nx_to_edge_index(G_nx, sorted_nodes):
    id2idx = {nid: i for i, nid in enumerate(sorted_nodes)}
    edges_mapped = []
    for u, v in G_nx.edges():
        if u in id2idx and v in id2idx:
            edges_mapped.append((id2idx[u], id2idx[v]))
            edges_mapped.append((id2idx[v], id2idx[u]))  # undirected
    if len(edges_mapped) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edges_mapped, dtype=torch.long).t().contiguous()
    return edge_index

def main(input_file='file.txt', device='cpu', learning_rate=0.01):
    device = torch.device(device)
    sorted_nodes, features, G_nx, feat_runtime = graph_to_adj_bet_return_scores(input_file)

    # Build edge_index
    edge_index = nx_to_edge_index(G_nx, sorted_nodes)
    if edge_index.numel() == 0:
        print("[-] Warning: edge_index is empty. Check your edgelist / node mapping.")

    features = features.to(device)
    edge_index = edge_index.to(device)

    # Model
    model = GraphSAGE3(
        in_channels=features.shape[1], hidden1=64, hidden2=32, out_channels=1
    ).to(device)

    # Add optimizer (even if no training happens)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Forward pass only (unsupervised — no training loop)
    eval_start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)  # shape (N,1)
        embeddings = out.view(-1)  # shape (N,)
    eval_runtime = time.time() - eval_start
    print(f"[+] Embedding forward runtime: {eval_runtime:.4f} s")

    # Save embeddings with node IDs
    embeddings_cpu = embeddings.cpu().numpy().tolist()
    node_embedding_pairs = [(nid, float(emb)) for nid, emb in zip(sorted_nodes, embeddings_cpu)]

    # Rank nodes by embedding value
    ranked = sorted(node_embedding_pairs, key=lambda x: x[1], reverse=True)
    rankings_out = [{"node": int(n), "embedding": s, "rank": i+1} for i, (n, s) in enumerate(ranked)]
    rankings_dict = {str(entry["node"]): entry["embedding"] for entry in rankings_out}

    # Save to file
    with open('Pathsage.txt', 'w') as fo:
        json.dump(rankings_dict, fo, indent=4)

    # Print as single JSON object
    print(json.dumps(rankings_dict, indent=4))

    return rankings_out
if __name__ == "__main__":
    G = nx.Graph()  # or use DiGraph() if your connections are directed
    rankings = main(input_file='file.txt', device='cpu')
