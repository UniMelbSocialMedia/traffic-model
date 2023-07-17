import numpy as np
import pandas as pd
import yaml

from data_loader.data_loader import DataLoader
from utils.dijkstra_algo import DijGraph

if __name__ == '__main__':
    # load configs
    with open("config/config.yaml", "r") as stream:
        configs = yaml.safe_load(stream)

        adj_filename = configs['adj_filename'] if configs['adj_filename'] else 'data/PEMS04/PEMS04.csv'
        graph_signal_matrix_filename = configs['graph_signal_matrix_filename'] if configs[
            'graph_signal_matrix_filename'] \
            else 'data/PEMS04/PEMS04.npz'
        pdf_hops_info_file = configs['graph_signal_matrix_filename_pdformer']
        num_of_vertices = configs['num_of_vertices'] if configs['num_of_vertices'] else 307

    hops_info = np.load(pdf_hops_info_file)

    edge_weights = np.zeros((num_of_vertices, num_of_vertices))
    w_info = pd.read_csv(adj_filename, header=None).values[1:]
    for row in range(w_info.shape[0]):
        v1 = int(w_info[row][0])
        v2 = int(w_info[row][1])
        w = float(w_info[row][2])

        edge_weights[v1, v2] = w
        edge_weights[v2, v1] = w

    dij = DijGraph(num_of_vertices)
    dij.graph = edge_weights

    f_out = open('./data/PEMS08/PEMS08_dij.csv', "w")
    f_out.write("from,to,cost\n")

    for v in range(num_of_vertices):
        dists = np.array(dij.dijkstra(v))
        filtered_dists = []
        for i, dist in enumerate(dists):
            if i == v: continue
            if dist <= 1000:
                filtered_dists.append(f'{str(v)},{str(i)},{str(dist)}\n')
                f_out.write(f'{str(v)},{str(i)},{str(dist)}\n')

        print(filtered_dists)

    f_out.close()
