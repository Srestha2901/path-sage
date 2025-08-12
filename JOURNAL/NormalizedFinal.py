import networkx as nx

data = []
seen_edges = set()

with open('HighSchool2013.txt', 'r') as file:
    for line in file:
        row = line.strip().split('\t')
        if len(row) >= 3:
            try:
                i = int(row[0])
                j = int(row[1])
                t = int(row[2])
                # Undirected: store min(i, j), max(i, j)
                edge_key = tuple(sorted((i, j)))  # For undirected uniqueness

                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    data.append((i, j, t))
            except ValueError:
                print(f"Skipping invalid line: {row}")
        else:
            print(f"Ignoring row: {row}. It doesn't have enough columns.")

# Create Graph and add edges with original timestamp
G = nx.Graph()  # or use DiGraph() if your connections are directed

for i, j, t in data:
    G.add_edge(i, j, time=t)

# Output info
print("Total number of nodes:", G.number_of_nodes())
print("Total number of edges:", G.number_of_edges())

print("Nodes:")
print(list(G.nodes))

# Write to input.txt
with open('input1.txt', 'w') as file:
    for u, v, attr in G.edges(data=True):
        file.write(f"{u}\t{v}\t{attr['time']}\n")

print("Edges in i j t format:")
for u, v, attr in G.edges(data=True):
    print(f"{u} {v} {attr['time']}")