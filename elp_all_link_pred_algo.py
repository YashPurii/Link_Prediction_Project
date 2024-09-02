import networkx as nx
import numpy as np
import math
import torch
import torch.nn as nn

# Transformer model definition
class TransformerLinkPredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads=8, n_layers=2, dropout=0.1):
        super(TransformerLinkPredictionModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return self.sigmoid(x)

def train_transformer_model(model, X_train, y_train, epochs=10, batch_size=32, learning_rate=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    return model

def transformer_link_prediction(adj, model):
    X, _ = prepare_transformer_data(adj)
    model.eval()
    with torch.no_grad():
        predictions = model(X).squeeze().numpy()
    return predictions.reshape(adj.shape)

def prepare_transformer_data(adj, sequence_length=5):
    sequences = []
    labels = []
    num_nodes = len(adj)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and adj[i, j] == 1:
                current_sequence = []
                for t in range(sequence_length):
                    if i + t + 1 < num_nodes and adj[i + t, j] == 1:
                        current_sequence.append(1)
                    else:
                        current_sequence.append(0)
                sequences.append(current_sequence)
                if i + sequence_length < num_nodes and adj[i + sequence_length, j] == 1:
                    labels.append(1)
                else:
                    labels.append(0)

    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.float32)

    return X, y

# Normalization function
def normalize(n):
    max_value = np.max(n)
    min_value = np.min(n)
    if max_value > 0:
        n = n / max_value
    return n

# Link prediction algorithms
def aa(adj):
    G = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for node1 in G:
        for node2 in G:
            common_neighbours_all = nx.common_neighbors(G, node1, node2)
            for common_neighbour in common_neighbours_all:
                if G.degree(common_neighbour) > 1:
                    common[node1][node2] += 1 / math.log(G.degree(common_neighbour))
    return common

def car(adj):
    Graph = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common_n = nx.common_neighbors(Graph, i, j)
            cn = len(sorted(nx.common_neighbors(Graph, i, j)))
            edges = 0
            if cn > 0:
                for m in common_n:
                    for n in common_n:
                        edges += adj[m, n]
            edges /= 2
            if edges > 0:
                common[i][j] += cn * (edges - 1)
    return common

def cclp(adj):
    Graph = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    triangles = nx.triangles(Graph)
    for i in range(len(adj)):
        for j in range(len(adj)):
            common_n = nx.common_neighbors(Graph, i, j)
            if len(sorted(nx.common_neighbors(Graph, i, j))) > 0:
                for k in common_n:
                    if Graph.degree(k) > 1:
                        common[i][j] += triangles[k] / (Graph.degree(k) * (Graph.degree(k) - 1) / 2)
    return common

def cclp2(adj):
    Graph = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    triangles = nx.triangles(Graph)
    for i in range(len(adj)):
        for j in range(len(adj)):
            common_n = nx.common_neighbors(Graph, i, j)
            for m in common_n:
                common_n2 = nx.common_neighbors(Graph, i, m)
                if len(sorted(nx.common_neighbors(Graph, i, m))) > 0:
                    for k in common_n2:
                        if Graph.degree(k) > 1:
                            common[i][j] += triangles[k] / (Graph.degree(k) * (Graph.degree(k) - 1) / 2)
    return common

def cn(adj):
    Graph = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            common[i][j] = len(sorted(nx.common_neighbors(Graph, i, j)))
    return common

def jc(adj):
    Graph = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for i in range(len(adj)):
        for j in range(len(adj)):
            n1 = list(Graph.neighbors(i))
            n2 = list(Graph.neighbors(j))
            length = len(set().union(n1, n2))
            if length > 0:
                common[i][j] = len(sorted(nx.common_neighbors(Graph, i, j))) / length
    return common

def shortest_path_lengths(adj):
    Graph = nx.Graph(adj)
    num_nodes = len(adj)
    shortest_paths = np.zeros((num_nodes, num_nodes), dtype=float)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            try:
                shortest_path_length = nx.shortest_path_length(Graph, source=i, target=j)
                shortest_paths[i][j] = shortest_paths[j][i] = shortest_path_length
            except nx.NetworkXNoPath:
                shortest_paths[i][j] = shortest_paths[j][i] = float('inf')
    
    return shortest_paths

def cosp(adj):
    Graph = nx.Graph(adj)
    num_nodes = len(adj)
    cos_similarity = np.zeros((num_nodes, num_nodes), dtype=float)
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            common_neighbors = len(list(nx.common_neighbors(Graph, i, j)))
            degree_i = len(list(Graph.neighbors(i)))
            degree_j = len(list(Graph.neighbors(j)))
            
            if degree_i != 0 and degree_j != 0:
                cosine_sim = common_neighbors / np.sqrt(degree_i * degree_j)
                cos_similarity[i][j] = cos_similarity[j][i] = cosine_sim
    
    return cos_similarity

def nlc(adj):
    Graph = nx.Graph(adj)
    triangles = nx.triangles(Graph)
    common = np.zeros((len(adj), len(adj)))
    for i in range(len(Graph)):
        for j in range(len(Graph)):
            if i != j:
                common_n = nx.common_neighbors(Graph, i, j)
                if len(sorted(nx.common_neighbors(Graph, i, j))) > 0:
                    for k in common_n:
                        cnxz = len(sorted(nx.common_neighbors(Graph, i, k)))
                        cnyz = len(sorted(nx.common_neighbors(Graph, j, k)))
                        kz = Graph.degree(k)
                        cz = triangles[k] / (kz * (kz - 1) / 2)
                        common[i][j] += (cnxz * cz / (kz - 1)) + (cnyz * cz / (kz - 1))
    return common

def pa(adj):
    g_train = nx.Graph(adj)
    common = np.zeros(adj.shape)
    for i in range(len(g_train)):
        for j in range(len(g_train)):
            common[i][j] = len(sorted(g_train.neighbors(i))) * len(sorted(g_train.neighbors(j)))
    return common

def ra(adj):
    G = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for node1 in G:
        for node2 in G:
            common_neighbours_all = nx.common_neighbors(G, node1, node2)
            for common_neighbour in common_neighbours_all:
                if G.degree(common_neighbour) != 0:
                    common[node1][node2] += 1 / G.degree(common_neighbour)
    return common

def rooted_pagerank_linkpred(adj):
    scores = []
    G = nx.Graph(adj)
    common = np.zeros((len(adj), len(adj)))
    for node1 in G:
        rooted_pagerank = nx.pagerank(G, personalization={node1: 1})
        for node2 in range(len(adj)):
            common[node1][node2] = rooted_pagerank[node2]
    return common

def clp_id_main(adj, tao, theta):
    G = nx.Graph(adj)
    var_dict = {'graph': G, 'tao': tao, 'theta': theta}
    for (u, v) in G.edges():
        value = random.uniform(0, 1)
        G.edges[u, v]['weight'] = value

    cluster_matrix = clustering_ss(G, var_dict)
    similarity_matrix = np.zeros((len(adj), len(adj)))
    overall_similarity_matrix = np.zeros((len(adj), len(adj)))

    for node1 in G:
        for node2 in G:
            similarity_matrix[node1][node2] = 1
            common_neighbour_factor = 1
            for node_neighbour in G.neighbors(node1):
                if G.has_edge(node2, node_neighbour):
                    common_neighbour_factor *= (1 - G.edges[node2, node_neighbour]['weight'])
            neighbour_factor = G.edges[node1, node2]['weight'] if G.has_edge(node1, node2) else 0
            similarity_matrix[node1][node2] = 1 - common_neighbour_factor + neighbour_factor

    similarity_matrix = normalize(similarity_matrix)

    for i in range(len(adj)):
        for j in range(len(adj)):
            overall_similarity_matrix[i][j] = similarity_matrix[i][j] * cluster_matrix[i][j]

    link_pred = np.zeros((len(adj), len(adj)))
    for node1 in G:
        for node2 in G:
            node_neighbour_common = nx.common_neighbors(G, node1, node2)
            for common_node in node_neighbour_common:
                link_pred[node1][node2] += overall_similarity_matrix[node1][common_node] + overall_similarity_matrix[common_node][node2]
    return link_pred

def clustering_ss(graph, var_dict):
    A = np.zeros((len(graph.nodes), len(graph.nodes)))
    var_dict['influence'] = A

    for node in graph:
        var_dict[node] = node
        for node_neighbour in graph.neighbors(node):
            if graph.edges[node, node_neighbour]['weight'] > random.uniform(0, 1):
                A[node][node_neighbour] = 1
            else:
                A[node][node_neighbour] = 0

    tao = var_dict['tao']
    i = 0
    while i <= tao:
        for node in graph:
            new_label = max_comm_label(node, graph, var_dict)
            var_dict[node] = new_label
        i += 1

    all_labels = set(var_dict[node] for node in graph)
    nodes_per_label = {label: sum(1 for node in graph if var_dict[node] == label) for label in all_labels}
    cluster_matrix = np.zeros((len(graph.nodes), len(graph.nodes)))

    for i in range(len(graph.nodes)):
        for j in range(len(graph.nodes)):
            if var_dict[i] == var_dict[j]:
                cluster_matrix[i][j] = nodes_per_label[var_dict[i]] / len(graph.nodes)
            else:
                cluster_matrix[i][j] = -nodes_per_label[var_dict[i]] / len(graph.nodes)

    return normalize(cluster_matrix)

def max_comm_label(node, graph, var_dict):
    all_labels = set(var_dict[node_neighbour] for node_neighbour in graph.neighbors(node))
    prob_actual = 1
    label_actual = var_dict[node]

    for label in all_labels:
        prob_new = 1
        for node_chk in graph.neighbors(node):
            if var_dict[node_chk] == label:
                chk = graph[node][node_chk]['weight'] if graph.has_edge(node, node_chk) else 0
                if var_dict['influence'][node][node_chk] == 1:
                    prob_new *= (1 - chk)
        if prob_new < prob_actual:
            prob_actual = prob_new
            label_actual = label
            var_dict[node] = label
    return label_actual
