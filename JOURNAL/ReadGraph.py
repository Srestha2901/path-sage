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


