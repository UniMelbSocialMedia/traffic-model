import pandas as pd
import numpy as np
import pickle

import torch

from utils.data_utils import scale_weights, attach_prev_dys_seq, seq_gen_v2, derive_rep_timeline
from data_loader.dataset import Dataset
from utils.math_utils import z_score_normalize, min_max_normalize


class DataLoader:
    def __init__(self, data_configs):
        self.dataset = None
        self.n_batch_train = None
        self.n_batch_test = None
        self.n_batch_val = None

        self.num_of_vertices = data_configs['num_of_vertices']
        self.points_per_hour = data_configs['points_per_hour']
        self.len_input = data_configs['len_input']
        self.last_day = data_configs['last_day']
        self.last_week = data_configs['last_week']
        self.num_days_per_week = data_configs['num_days_per_week']
        self.rep_vectors = data_configs['rep_vectors']

        self.distance_threshold = data_configs['distance_threshold']

        self.batch_size = data_configs['batch_size']
        self.enc_features = data_configs['enc_features']
        self.dec_seq_offset = data_configs['dec_seq_offset']

        self.preprocess = data_configs['preprocess']
        self.preprocess_output_path = data_configs['preprocess_output_path']
        self.node_data_filename = data_configs['node_data_filename']

        self.edge_weight_filename = data_configs['edge_weight_filename']
        self.semantic_adj_filename = data_configs['semantic_adj_filename']
        self.edge_weight_original_filename = data_configs['edge_weight_original_filename']
        self.edge_weight_scaling = data_configs['edge_weight_scaling']

        # PEMSD7 Specific Variables
        self.n_train = 34
        self.n_test = 5
        self.n_val = 5
        self.day_slot = self.points_per_hour * 24
        self.n_seq = self.len_input * 2
        self.points_per_week = self.points_per_hour * 24 * self.num_days_per_week

    def _generate_new_x_arr(self, x_set: np.array, records_time_idx: dict):
        # WARNING: This has be changed accordingly.
        speed_idx = 0
        last_dy_idx = 2
        last_wk_idx = 3

        # Attach rep vectors for last day and last week data and drop weekly time index value
        new_n_f = x_set.shape[3] - 1
        # To add rep last hour seq
        if self.rep_vectors:
            new_n_f += 1
        # To add rep last dy seq
        if self.last_day and self.rep_vectors:
            new_n_f += 1
        # To add rep last wk seq
        if self.last_week and self.rep_vectors:
            new_n_f += 1

        new_x_set = np.zeros((x_set.shape[0], x_set.shape[1], x_set.shape[2], new_n_f))
        for i, x in enumerate(x_set):
            # WARNING: had to determine which index represent weekly time idx
            record_key = x[0, 0, 1]
            record_key_yesterday = record_key - 24 * self.points_per_hour
            if record_key_yesterday < 0: record_key_yesterday = record_key + self.points_per_week - 24 * self.points_per_hour

            tmp = x[:, :, speed_idx:speed_idx + 1]
            if self.last_day:
                last_dy_data = x[:, :, last_dy_idx:last_dy_idx + 1]
                tmp = np.concatenate((tmp, last_dy_data), axis=-1)
            if self.last_week:
                last_wk_data = x[:, :, last_wk_idx:last_wk_idx + 1]
                tmp = np.concatenate((tmp, last_wk_data), axis=-1)
            if self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)
            if self.last_day and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key_yesterday]), axis=-1)
            if self.last_week and self.rep_vectors:
                tmp = np.concatenate((tmp, records_time_idx[record_key]), axis=-1)

            new_x_set[i] = tmp

        time_idx = x_set[:, :, :, 1:2]
        return new_x_set, time_idx

    # generate training, validation and test data
    def load_node_data_file(self):
        if not self.preprocess:
            preprocessed_file = open(self.preprocess_output_path, 'rb')
            self.dataset = pickle.load(preprocessed_file)
            return

        train_data = np.load('data/METRLA/train.npz')
        test_data = np.load('data/METRLA/test.npz')
        val_data = np.load('data/METRLA/val.npz')
        new_train_x_set, training_y_set = train_data['x'], train_data['y']
        new_test_x_set, testing_y_set = test_data['x'], test_data['y']
        new_val_x_set, validation_y_set = val_data['x'], val_data['y']

        # Add tailing target values form x values to facilitate local trend attention in decoder
        training_y_set = np.concatenate(
            (new_train_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], training_y_set[:, :, :, 0:1]), axis=1)
        validation_y_set = np.concatenate(
            (new_val_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], validation_y_set[:, :, :, 0:1]), axis=1)
        testing_y_set = np.concatenate(
            (new_test_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], testing_y_set[:, :, :, 0:1]), axis=1)

        # z-score normalization on input and target values
        stats_x, x_train, x_val, x_test = min_max_normalize(new_train_x_set, new_val_x_set, new_test_x_set)
        stats_y, y_train, y_val, y_test = min_max_normalize(training_y_set, validation_y_set, testing_y_set)

        # shuffling training data 0th axis
        idx_samples = np.arange(0, x_train.shape[0])
        np.random.shuffle(idx_samples)
        x_train = x_train[idx_samples]
        y_train = y_train[idx_samples]

        self.n_batch_train = int(len(x_train) / self.batch_size)
        self.n_batch_test = int(len(x_test) / self.batch_size)
        self.n_batch_val = int(len(x_val) / self.batch_size)

        data = {'train': x_train, 'val': x_val, 'test': x_test}
        y = {'train': y_train[:, :, :, 0:1], 'val': y_val[:, :, :, 0:1], 'test': y_test[:, :, :, 0:1]}
        self.dataset = Dataset(
            data=data,
            y=y,
            stats_x=stats_x,
            stats_y=stats_y,
            n_batch_train=self.n_batch_train,
            n_batch_test=self.n_batch_test,
            n_batch_val=self.n_batch_val,
            batch_size=self.batch_size,
        )

        with open(self.preprocess_output_path, 'wb') as file:
            pickle.dump(self.dataset, file)

    def get_dataset(self):
        return self.dataset

    def load_edge_data_file(self):
        try:
            w = pd.read_csv(self.edge_weight_filename, header=None).values[1:]

            dst_edges = []
            src_edges = []
            edge_attr = []
            for row in range(w.shape[0]):
                # Drop edges with large distance between vertices. This adds incorrect attention in training time and
                # degrade test performance (Over-fitting).
                if float(w[row][2]) > self.distance_threshold:
                    continue
                dst_edges.append(int(float(w[row][0])))
                src_edges.append(int(float(w[row][1])))
                edge_attr.append([float(w[row][2])])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(np.array(edge_attr), True, min_max=True)

            return edge_index, edge_attr

        except FileNotFoundError:
            print(f'ERROR: input file was not found in {self.edge_weight_filename}.')

    def load_semantic_edge_data_file(self):
        # semantic_file = open(self.semantic_adj_filename, 'rb')
        # sensor_details = pickle.load(semantic_file)

        dst_edges = []
        src_edges = []
        edge_attr = []
        # for i, (sensor, neighbours) in enumerate(sensor_details.items()):
        #     for j, (neighbour, distance) in enumerate(neighbours.items()):
        #         if j > 2:
        #             break
        #         dst_edges.append(sensor)
        #         src_edges.append(neighbour)
        #         edge_attr.append([distance])
        #
        edge_index = [src_edges, dst_edges]
        # edge_attr = scale_weights(edge_attr, self.edge_weight_scaling, min_max=True)

        return edge_index, edge_attr

    def load_batch(self, _type: str, offset: int, device: str = 'cpu'):
        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)
        limit = (offset + self.batch_size) if (offset + self.batch_size) <= len(xs) else len(xs)

        enc_xs_time_idx = None
        xs = xs[offset: limit, :, :, :]  # [9358, 13, 228, 1] # Avoid selecting time idx
        ys = ys[offset: limit, :]

        # ys_input will be used as decoder inputs while ys will be used as ground truth data
        ys_input = np.copy(ys)
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, :] = 0

        num_inner_f_enc = int(xs.shape[-1] / self.enc_features)
        enc_xs = [torch.Tensor(xs).to(device)]

        dec_ys = [[] for i in range(self.batch_size)]  # decoder input
        dec_ys_target = [[] for i in range(self.batch_size)]  # This is used as the ground truth data
        for idx, y_timesteps in enumerate(ys_input):
            dec_ys[idx] = torch.Tensor(y_timesteps).to(device)
            dec_ys_target[idx] = torch.Tensor(ys[idx]).to(device)

        dec_ys = torch.stack(dec_ys)

        return enc_xs, enc_xs_time_idx, dec_ys, dec_ys_target
