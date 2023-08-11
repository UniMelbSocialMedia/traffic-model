import pandas as pd
import numpy as np
from dtw import *
import pickle

from utils.data_utils import seq_gen_v2, derive_rep_timeline, scale_weights


def load_rep_vector(node_data_filename, output_filename, load_file=False):
    if load_file:
        output_file = open(output_filename, 'rb')
        records_time_idx = pickle.load(output_file)
        return records_time_idx

    points_per_hour = 12
    len_input = 12
    n_train = 34
    n_test = 5
    n_val = 5
    num_days_per_week = 5
    day_slot = points_per_hour * 24
    n_seq = len_input * 2
    points_per_week = points_per_hour * 24 * num_days_per_week
    num_of_vertices = 228

    data_seq = pd.read_csv(node_data_filename, header=None).values

    total_days = n_train + n_test + n_val
    seq_train = seq_gen_v2(n_train, data_seq, 0, n_seq, num_of_vertices, day_slot, 1, total_days)

    # Take seq all to find last day, last week time seq
    new_seq_all = np.zeros((seq_train.shape[0], seq_train.shape[1], seq_train.shape[2], seq_train.shape[3] + 1))

    for idx in range(seq_train.shape[0]):
        time_idx = np.array(idx % points_per_week)
        new_arr = np.expand_dims(np.reshape(np.repeat(time_idx, n_seq * num_of_vertices, axis=0),
                                            (n_seq, num_of_vertices)), axis=2)
        new_seq_all[idx] = np.concatenate((seq_train[idx], new_arr), axis=-1)

    # Derive global representation vector for each sensor for similar time steps
    records_time_idx = derive_rep_timeline(new_seq_all[:, :len_input], day_slot * num_days_per_week, num_of_vertices)

    with open(output_filename, 'wb') as file:
        pickle.dump(records_time_idx, file)

    return records_time_idx


def set_edge_semantics(time_idx_file):
    output_file = open(time_idx_file, 'rb')
    time_idx_semantics_rels = pickle.load(output_file)

    time_idx_edge_details = {}
    for i, (time_idx, semantic_rels) in enumerate(time_idx_semantics_rels.items()):
        dst_edges = []
        src_edges = []
        edge_attr = []
        for i, (sensor, neighbours) in enumerate(semantic_rels.items()):
            for j, (neighbour, distance) in enumerate(neighbours.items()):
                if j > 2:
                    break
                dst_edges.append(sensor)
                src_edges.append(neighbour)
                edge_attr.append([distance])

        edge_index = [src_edges, dst_edges]
        edge_attr = scale_weights(edge_attr, scaling=True, min_max=True)

        time_idx_edge_details[time_idx] = (edge_index, edge_attr)

    return time_idx_edge_details


def load_data_seq(filename):
        data_seq = pd.read_csv(filename, header=None).values
        len = data_seq.shape[0]
        last_month_seq = data_seq[int(len/2):]

        return last_month_seq


if __name__ == '__main__':

    graph_signal_matrix_filename = "../data/PEMSD7/PeMSD7_V_228.csv"
    rep_output_file = "../data/PEMSD7/PeMSD7_rep_vector.csv"
    time_idx_rep_output_file = "../data/PEMSD7/PeMSD7_time_idx_semantic_rels.pickle"
    time_idx_edge_details_file = "../data/PEMSD7/PeMSD7_time_idx_semantic_edges.pickle"
    # records_time_idx = load_rep_vector(graph_signal_matrix_filename, rep_output_file, load_file=True)
    #
    # n_sensors = 228
    # time_idx_semantic_rels = {}
    # for i, (time_idx, time_series) in enumerate(records_time_idx.items()):
    #     print(f"Time idx: {time_idx}")
    #     semantic_rels = {}
    #     for sensor in range(n_sensors):
    #         print(f"Processing Sensor: {sensor}")
    #         sensor_seq = time_series[:, sensor]
    #         alignment_details = []
    #         distances = []
    #
    #         for sensor_2 in range(n_sensors):
    #             if sensor_2 == sensor: continue
    #             sensor_seq_2 = time_series[:, sensor_2]
    #             try:
    #                 alignment = dtw(sensor_seq, sensor_seq_2, window_type="sakoechiba", window_args={'window_size': 3})
    #                 alignment_details.append(alignment)
    #                 distances.append(alignment.distance)
    #             except ValueError as ex:
    #                 print(ex)
    #
    #         min_indices = np.argpartition(distances, 5)[:5]
    #         sorted_distances = np.array(distances)[min_indices]
    #         min_data = {}
    #         for i, min_idx in enumerate(min_indices):
    #             min_data[min_idx] = sorted_distances[i]
    #         semantic_rels[sensor] = min_data
    #
    #     time_idx_semantic_rels[time_idx] = semantic_rels
    #
    # with open(time_idx_rep_output_file, 'wb') as file:
    #     pickle.dump(time_idx_semantic_rels, file)

    time_idx_edge_details = set_edge_semantics(time_idx_rep_output_file)
    with open(time_idx_edge_details_file, 'wb') as file:
        pickle.dump(time_idx_edge_details, file)

    # n_sensors = 228
    #
    # data_seq = load_data_seq(graph_signal_matrix_filename)
    #
    # semantic_rels = {}
    # for sensor in range(n_sensors):
    #     print("Processing Sensor: {}".format(sensor))
    #     sensor_seq = data_seq[:, sensor]
    #     alignment_details = []
    #     distances = []
    #
    #     for sensor_2 in range(n_sensors):
    #         if sensor_2 == sensor: continue
    #         sensor_seq_2 = data_seq[:, sensor_2]
    #         try:
    #             alignment = dtw(sensor_seq, sensor_seq_2, window_type="sakoechiba", window_args={'window_size': 3})
    #             alignment_details.append(alignment)
    #             distances.append(alignment.distance)
    #         except ValueError as ex:
    #             print(ex)
    #
    #     min_indices = np.argpartition(distances, 5)[:5]
    #     sorted_distances = np.array(distances)[min_indices]
    #     min_data = {}
    #     for i, min_idx in enumerate(min_indices):
    #         min_data[min_idx] = sorted_distances[i]
    #     semantic_rels[sensor] = min_data
    #     # for idx in min_indices:
    #     #     alignment = alignment_details[idx]
    #     #     plt.plot(data_seq[:, idx])
    #     #     plt.plot(alignment.index2, sensor_seq[alignment.index1])
    #     #     plt.show()
    #     #     plt.close()
    #
    # with open("../data/PEMSD7/PeMSD7_W_228.pickle", 'wb') as file:
    #     pickle.dump(semantic_rels, file)


