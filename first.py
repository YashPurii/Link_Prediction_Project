import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, \
    precision_score, f1_score, precision_recall_curve, accuracy_score, balanced_accuracy_score

from elp_all_link_pred_algo import aa, ra, cclp, cclp2, cn, pa, jc, car, \
    rooted_pagerank_linkpred, normalize, clp_id_main, elp, nlc, cosp

from comm_dyn import features_comm_opti, local_path
from sklearn.model_selection import train_test_split

import time
import random
from xlwt import Workbook
import xlrd
import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from elp_all_link_pred_algo import TransformerLinkPredictionModel

if __name__ == '__main__':
    starttime_full = time.time()
    var_dict_main = {}

    def auprgraph_all (adj, file_name, algo):
        file_write_name = './result_all_elp/result_' + algo + '/' + file_name + ".txt"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        starttime_aup = time.time()
        ratio = []
        aupr = []
        auc = []
        avg_prec = []
        G = nx.Graph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
        for i in [0.7, 0.8, 0.9]:
            print("nodes - " + str(len(adj)) + " edges - " + str(G.number_of_edges()) + " name - " + str(file_name))
            print("For ratio : ", i-1)
            if algo in ["cn", "aa", "jc", "pa", "cclp", "mfi", "elp", "nlc", "cosp", "ra", "l3", "act", "car", "local_path", "sp"]:
                print("algo - " + algo)
                avg_array = avg_seq_all(G, file_name, i, algo)
            elif algo in ["feature_comm_opti"]:
                avg_array = avg_seq(G, file_name, i, algo)
            aupr.append(avg_array[0])
            auc.append(avg_array[1])
            avg_prec.append(avg_array[2])
            ratio.append(i-1)
        print("Ratio:-", ratio)
        print("AUPR:-", aupr)
        print("AUC:-", auc)
        print("Avg Precision:-", avg_prec)
        endtime_aup = time.time()
        print('That aup took {} seconds'.format(endtime_aup - starttime_aup))

        wb = Workbook()
        sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)
        sheet1.write(0, 0, 'Ratio')
        sheet1.write(0, 1, 'AUPR')
        sheet1.write(0, 2, 'AUC')
        sheet1.write(0, 3, 'AVG PRECISION')
        for i in range(3):
            sheet1.write(3 - i, 0, ratio[i]*-1)
            sheet1.write(3 - i, 1, aupr[i])
            sheet1.write(3 - i, 2, auc[i])
            sheet1.write(3 - i, 3, avg_prec[i])

        wb.save('./result_all_elp/result_' + algo + '/' + file_name + ".xls")

        currentDT = datetime.datetime.now()
        print(str(currentDT))

        file_all = open('./result_all_elp/current_all.txt', 'a')
        text_final = "full algo = " + algo + " file name = " + file_name + " time = " + \
                     str((endtime_aup - starttime_aup)) + " date_time = " + str(currentDT) + "\n"
        file_all.write(text_final)
        print(text_final)
        file_all.close()

        return aupr, ratio, auc, avg_prec

    def avg_seq(g, file_name, ratio, algo):
        start_time_ratio = time.time()
        aupr = 0
        auc = 0
        avg_prec = 0
        loop = 50
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
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
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            if algo == 'feature_comm_opti':
                m = 5
                t = "fb-forum"
                prob_mat = features_comm_opti(m, t)

            prob_mat = normalize(prob_mat)
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            adj_test = nx.adjacency_matrix(graph_test).todense()
            array_true = []
            array_pred = []
            rows = prob_mat.shape[0]
            cols = prob_mat.shape[1]
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if rows > i and cols > j:
                        if not graph_original.has_edge(i, j):
                            array_true.append(0)
                            array_pred.append(prob_mat[i][j])
                        if graph_test.has_edge(i, j):
                            array_true.append(1)
                            array_pred.append(prob_mat[i][j])

            pred = array_pred
            adj_test = array_true

            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            auc += auc_value
            avg_prec += avg_prec_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./result_all_elp/current.txt', 'a')
        text_inside = "full algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " time = " + \
                           str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, auc / loop, avg_prec / loop

    def avg_seq_all(g, file_name, ratio, algo):
        start_time_ratio = time.time()
        aupr = 0
        auc = 0
        avg_prec = 0
        loop = 50
        ratio = round(ratio, 1)
        graph_original = g
        print("avg sequential called for algo - " + str(algo) + " ratio - " + str(ratio))

        for single_iter in range(loop):
            print("old number of edges - " + str(len(graph_original.edges)) + " for ratio - " + str(ratio))
            adj_original = nx.adjacency_matrix(graph_original).todense()
            starttime = time.time()
            edges = np.array(list(graph_original.edges))
            nodes = list(range(len(adj_original)))
            np.random.shuffle(edges)
            edges_original = edges
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
            print("new number of edges - " + str(len(graph_train.edges)) + " for ratio - " + str(ratio))

            if algo == 'cn': prob_mat = cn(adj_train)
            if algo == 'ra': prob_mat = ra(adj_train)
            if algo == 'car': prob_mat = car(adj_train)
            if algo == 'cclp': prob_mat = cclp(adj_train)
            if algo == 'jc': prob_mat = jc(adj_train)
            if algo == 'pa': prob_mat = pa(adj_train)
            if algo == 'aa': prob_mat = aa(adj_train)
            if algo == 'elp': prob_mat = elp(adj_train)
            if algo == 'mfi': prob_mat = clp_id_main(adj_train, 25, 1.0)
            if algo == 'nlc': prob_mat = nlc(adj_train)
            if algo == 'cosp': prob_mat = cosp(adj_train)
            if algo == 'act': prob_mat = cclp2(adj_train)  
            if algo == 'local_path': prob_mat = local_path(adj_train) 
            if algo == 'sp': prob_mat = rooted_pagerank_linkpred(adj_train)

            prob_mat = normalize(prob_mat)
            endtime = time.time()
            print('{} for probability matrix prediction'.format(endtime - starttime))

            adj_test = nx.adjacency_matrix(graph_test).todense()
            array_true = []
            array_pred = []
            for i in range(len(adj_original)):
                for j in range(len(adj_original)):
                    if not graph_original.has_edge(i, j):
                        array_true.append(0)
                        array_pred.append(prob_mat[i][j])
                    if graph_test.has_edge(i, j):
                        array_true.append(1)
                        array_pred.append(prob_mat[i][j])

            pred = array_pred
            adj_test = array_true

            prec_per, recall_per, threshold_per = precision_recall_curve(adj_test, pred)
            prec_per = prec_per[::-1]
            recall_per = recall_per[::-1]
            aupr_value = np.trapz(prec_per, x=recall_per)
            auc_value = roc_auc_score(adj_test, pred)
            avg_prec_value = average_precision_score(adj_test, pred)

            test_pred_label = np.copy(pred)
            a = np.mean(test_pred_label)

            for i in range(len(pred)):
                if pred[i] < a:
                    test_pred_label[i] = 0
                else:
                    test_pred_label[i] = 1
            recall_value = recall_score(adj_test, test_pred_label)
            acc_score_value = accuracy_score(adj_test, test_pred_label)
            bal_acc_score_value = balanced_accuracy_score(adj_test, test_pred_label)
            precision_value = precision_score(adj_test, test_pred_label)
            f1_value = f1_score(adj_test, test_pred_label)

            endtime = time.time()
            print('{} for metric calculation'.format(endtime - starttime))

            currentDT = datetime.datetime.now()
            print(str(currentDT))

            file_all = open('./result_all_elp/current.txt', 'a')
            text_inside_single = "single algo = " + algo + " file name = " + file_name + \
                                 " ratio = " + str(ratio) + " time = " + \
                                 str(endtime - starttime) + " sec date_time = " + str(currentDT) + "\n"
            file_all.write(text_inside_single)
            print(text_inside_single)
            file_all.close()

            aupr += aupr_value
            auc += auc_value
            avg_prec += avg_prec_value

        currentDT = datetime.datetime.now()
        print(str(currentDT))
        end_time_ratio = time.time()
        file_all = open('./result_all_elp/current.txt', 'a')
        text_inside = "full algo = " + algo + " file name = " + file_name + \
                           " ratio = " + str(ratio) + " time = " + \
                           str(end_time_ratio - start_time_ratio) + " date_time = " \
                           + str(currentDT) + "\n"
        file_all.write(text_inside)
        file_all.close()

        return aupr / loop, auc / loop, avg_prec / loop


    def create_transformer_model(input_dim, output_dim, n_heads=8, n_layers=2, dropout=0.1):
        class TransformerLinkPredictionModel(nn.Module):
            def __init__(self):
                super(TransformerLinkPredictionModel, self).__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=n_heads, dropout=dropout)
                self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
                self.fc = nn.Linear(input_dim, output_dim)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.transformer_encoder(x)
                x = self.fc(x)
                return self.sigmoid(x)
        
        return TransformerLinkPredictionModel()

    def train_transformer_model(model, X_train, y_train, epochs=10, batch_size=32, learning_rate=1e-4):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    def aupgraph_control_multiple_dataset_all(file_name_array):
        file_write_name = "./datasets_dynamic/data_info/current_all.txt"
        os.makedirs(os.path.dirname(file_write_name), exist_ok=True)
        algo_ego = ['feature_comm_opti']
        for algo in algo_ego:
            if algo not in []:
                for file_name in file_name_array:
                    ds = './datasets_dynamic/' + file_name
                    G = read_txt(ds + '.txt')
                    adj_mat_s = nx.adjacency_matrix(G)
                    n = adj_mat_s.shape[0]
                    print("nodes = " + str(n))
                    adj_mat_d = adj_mat_s.todense()
                    adj = adj_mat_d
                    auprgraph_all(adj, file_name, algo)

    algo_result_ego = ['feature_comm_opti']

    file_name_array = ['Eu-core']

    aupgraph_control_multiple_dataset_all(file_name_array)
    result_parser_combine(file_name_array, algo_result_ego)

    print('That took {} seconds'.format(time.time() - starttime_full))
