import pandas as pd
import numpy as np
import networkx as nx
import random
from comm_dyn import features_comm_opti
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from elp_all_link_pred_algo import normalize, TransformerLinkPredictionModel

if __name__ == '__main__':
    def grid_search_transformer(param_grid, X_train, y_train, X_test, y_test):
        best_score = -np.inf
        best_params = None

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=param_grid['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=param_grid['batch_size'], shuffle=False)

        for params in ParameterGrid(param_grid):
            print("Trying parameters:", params)

            model = TransformerLinkPredictionModel(input_dim=params['input_dim'], 
                                                   output_dim=1, 
                                                   n_heads=params['n_heads'], 
                                                   n_layers=params['n_layers'],
                                                   dropout=params['dropout'])
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.BCELoss()

            model.train()
            for epoch in range(params['epochs']):
                total_loss = 0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output.squeeze(), y_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}/{params['epochs']}, Loss: {total_loss/len(train_loader)}")

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    output = model(X_batch)
                    predicted = (output.squeeze() > 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)

            accuracy = correct / total

            if accuracy > best_score:
                best_score = accuracy
                best_params = params

        print("Best parameters found:", best_params)
        print("Best accuracy found:", best_score)

    G = nx.Graph()
    with open('fb-forum.txt', 'r') as file:
        for line in file:
            edge = line.strip().split()
            source = int(edge[0])
            target = int(edge[1])
            G.add_edge(source, target)

    graph_original = G
    adj_original = nx.adjacency_matrix(graph_original).todense()
    edges = np.array(list(graph_original.edges))
    nodes = list(range(len(adj_original)))
    np.random.shuffle(edges)
    edges_original = edges
    ratio = 0.8
    edges_train = np.array(edges_original, copy=True)
    np.random.shuffle(edges_train)
    edges_train = random.sample(list(edges_train), int(ratio * (len(edges_train))))
    graph_train = nx.Graph()
    graph_train.add_nodes_from(nodes)
    graph_train.add_edges_from(edges_train)
    adj_train = nx.adjacency_matrix(graph_train).todense()
    graph_test = nx.Graph()
    graph_test.add_nodes_from(nodes)
    graph_test.add_edges_from(edges_original)
    graph_test.remove_edges_from(edges_train)
    t = 'fb-forum'
    m = 5
    prob_mat = features_comm_opti(m, t)

    prob_mat = normalize(prob_mat)

    seq_length = 10

    sequences = []
    labels = []
    for i in range(0, prob_mat.shape[0] - seq_length + 1):
        sequences.append(prob_mat[i:i + seq_length, :])
        if graph_train.has_edge(i, i + seq_length - 1):
            labels.append(1)
        else:
            labels.append(0)
    sequences = np.array(sequences)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    param_grid = {
        'input_dim': [seq_length],
        'n_heads': [4, 8],
        'n_layers': [2, 4],
        'dropout': [0.1, 0.2],
        'learning_rate': [1e-4, 1e-3],
        'batch_size': [32, 64],
        'epochs': [5, 10, 15]
    }

    grid_search_transformer(param_grid, X_train, y_train, X_test, y_test)
