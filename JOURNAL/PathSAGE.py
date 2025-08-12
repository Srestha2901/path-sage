
# graphsage_rankings_unsupervised.py
import networkx as nx
import json
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

# ---------- Feature Extraction ----------
def graph_to_adj_bet_return_scores(graph_file):
    t0 = time.time()
    G = nx.read_weighted_edgelist(graph_file, nodetype=int, create_using=nx.MultiGraph())

    valid_paths_for_nodes = {}
    for node in G.nodes():
        valid_paths = []
        seen_first_middle = set()
        for neighbor1 in G.neighbors(node):
            for neighbor2 in G.neighbors(node):
                if neighbor1 == neighbor2:
                    continue
                edges1 = G.get_edge_data(node, neighbor1)
                edges2 = G.get_edge_data(node, neighbor2)
                if edges1 is None or edges2 is None:
                    continue

                for _, data1 in edges1.items():
                    for _, data2 in edges2.items():
                        weight1 = data1.get('weight', 0)
                        weight2 = data2.get('weight', 0)
                        if weight1 < weight2:
                            first_middle_pair = tuple(sorted((neighbor1, node)))
                            if first_middle_pair not in seen_first_middle:
                                seen_first_middle.add(first_middle_pair)
                                valid_paths.append([neighbor1, node, neighbor2])

        valid_paths_for_nodes[node] = valid_paths

    temporal_aggregated_scores = {node: len(paths) for node, paths in valid_paths_for_nodes.items()}

    # Save raw scores
    with open('Pathsage1.txt', 'w') as f:
        json.dump({str(k): int(v) for k, v in temporal_aggregated_scores.items()}, f, indent=4)

    sorted_nodes = sorted(temporal_aggregated_scores.keys())
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

# ---------- Main (unsupervised) ----------
def main(input_file='input1.txt', device='cpu'):
    device = torch.device(device)
    sorted_nodes, features, G_nx, feat_runtime = graph_to_adj_bet_return_scores(input_file)

    # Build edge_index
    edge_index = nx_to_edge_index(G_nx, sorted_nodes)
    if edge_index.numel() == 0:
        print("[-] Warning: edge_index is empty. Check your edgelist / node mapping.")

    features = features.to(device)
    edge_index = edge_index.to(device)

    # Model
    model = GraphSAGE3(in_channels=features.shape[1], hidden1=64, hidden2=32, out_channels=1).to(device)

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

    with open('node_rankings.txt', 'w') as fo:
        json.dump(rankings_out, fo, indent=4)

    # Show top 10
    print("\nTop 10 nodes by final embedding value:")
    for i, entry in enumerate(rankings_out[:10]):
        print(f" {i+1}. Node {entry['node']}  embedding={entry['embedding']:.6f}")

    return rankings_out

if __name__ == "__main__":
    rankings = main(input_file='input1.txt', device='cpu')

