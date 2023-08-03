import pandas as pd
import numpy as np
import pickle

import torch
from torch import Tensor
from torch_geometric.transforms import ToDevice
import torch_geometric.data as data

from utils.data_utils import scale_weights, attach_prev_dys_seq, seq_gen_v2, derive_rep_timeline
from data_loader.dataset import Dataset
from utils.math_utils import z_score_normalize


class DataLoader:
    def __init__(self, data_configs):
        self.dataset = None
        self.edge_index = None
        self.edge_attr = None
        self.edge_index_semantic = None
        self.edge_attr_semantic = None
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

        self.batch_size = data_configs['batch_size']
        self.graph_enc_input = data_configs['graph_enc_input']
        self.graph_dec_input = data_configs['graph_dec_input']
        self.non_graph_enc_input = data_configs['non_graph_enc_input']
        self.non_graph_dec_input = data_configs['non_graph_dec_input']
        self.enc_features = data_configs['enc_features']
        self.dec_seq_offset = data_configs['dec_seq_offset']

        # PEMSD7 Specific Variables
        self.n_train = 34
        self.n_test = 5
        self.n_val = 5
        self.day_slot = self.points_per_hour * 24
        self.n_seq = self.len_input * 2

    # generate training, validation and test data
    def load_node_data_file(self, filename: str, save=False):
        data_seq = pd.read_csv(filename, header=None).values

        total_days = self.n_train + self.n_test + self.n_val
        seq_train = seq_gen_v2(self.n_train, data_seq, 0, self.n_seq, self.num_of_vertices, self.day_slot, 1,
                               total_days)
        seq_val = seq_gen_v2(self.n_val, data_seq, self.n_train, self.n_seq, self.num_of_vertices, self.day_slot, 1,
                             total_days)
        seq_test = seq_gen_v2(self.n_test, data_seq, self.n_train + self.n_val, self.n_seq, self.num_of_vertices,
                              self.day_slot, 1, total_days)

        # Take seq all to find last day, last week time seq
        seq_all = np.concatenate((seq_train, seq_val, seq_test), axis=0)
        new_seq_all = np.zeros((seq_all.shape[0], seq_all.shape[1], seq_all.shape[2], seq_all.shape[3] + 1))
        points_per_week = self.points_per_hour * 24 * self.num_days_per_week
        for idx in range(seq_all.shape[0]):
            time_idx = np.array(idx % points_per_week)
            new_arr = np.expand_dims(np.reshape(np.repeat(time_idx, self.n_seq * self.num_of_vertices, axis=0),
                                                (self.n_seq, self.num_of_vertices)), axis=2)
            new_seq_all[idx] = np.concatenate((seq_all[idx], new_arr), axis=-1)

        # Following idx will be used to filter out the weekly time idx that we added in the prev step
        speed_idx = 0
        last_dy_idx = 2
        last_wk_idx = 3

        # attach last day and last week time series with last hour data
        # Warning: we attached weekly index along with the speed value in the prev step.
        # So, picking right index is important from now on. new_seq_all -> (8354, 12, 228, 1) -> (8354, 12, 228, 2)
        # In the following step we attach one or two values pertaining to last day and last week speed values
        x = attach_prev_dys_seq(new_seq_all,
                                self.len_input,
                                self.day_slot,
                                self.num_days_per_week,
                                self.n_train,
                                self.n_val,
                                self.last_week,
                                self.last_day)
        training_x_set, validation_x_set, testing_x_set = x['train'], x['val'], x['test']

        # Derive global representation vector for each sensor for similar time steps
        if self.rep_vectors:
            records_time_idx = derive_rep_timeline(training_x_set,
                                                   self.day_slot * self.num_days_per_week,
                                                   self.num_of_vertices)

        # avoided mixing training, testing, and validation dataset at the edge.
        # The time series during the last two hours of train, test and val datasets are ignored
        training_x_set = training_x_set[: -1 * self.n_seq]
        validation_x_set = validation_x_set[: -1 * self.n_seq]
        testing_x_set = testing_x_set[: -1 * self.n_seq]

        # When we consider last day or last week data, we have to drop a certain amount data in training
        # y dataset as done in training x dataset.
        total_drop = self.day_slot * 1
        training_y_set = seq_train[total_drop:-1 * self.n_seq, self.len_input:]
        validation_y_set = seq_val[:-1 * self.n_seq, self.len_input:]
        testing_y_set = seq_test[:-1 * self.n_seq, self.len_input:]

        # Attach rep vectors for last day and last week data and drop weekly time index value
        new_n_f = training_x_set.shape[3]-1
        if self.last_day and self.rep_vectors:
            new_n_f += 1
        if self.last_week and self.rep_vectors:
            new_n_f += 1

        new_train_x_set = np.zeros((training_x_set.shape[0], training_x_set.shape[1], training_x_set.shape[2], new_n_f))
        for i, x in enumerate(training_x_set):
            record_key = x[0, 0, -2]
            record_key_yesterday = record_key - 24 * self.points_per_hour
            record_key_yesterday = record_key_yesterday if record_key_yesterday >= 0 else record_key + points_per_week - 24 * self.points_per_hour

            speed_data = x[:, :, speed_idx:speed_idx + 1]
            last_dy_data = x[:, :, last_dy_idx:last_dy_idx + 1]
            last_wk_data = x[:, :, last_wk_idx:last_wk_idx + 1]
            x = np.concatenate(
                (speed_data, last_dy_data, last_wk_data, records_time_idx[record_key], records_time_idx[record_key_yesterday]),
                axis=-1)
            new_train_x_set[i] = x

        new_val_x_set = np.zeros((validation_x_set.shape[0], validation_x_set.shape[1], validation_x_set.shape[2],
                                  new_n_f))
        for i, x in enumerate(validation_x_set):
            record_key = x[0, 0, -2]
            record_key_yesterday = record_key - 24 * 12
            record_key_yesterday = record_key_yesterday if record_key_yesterday >= 0 else record_key + points_per_week - 24 * 12
            x = np.concatenate(
                (x[:, :, 0:1], x[:, :, 2:3], records_time_idx[record_key], records_time_idx[record_key_yesterday]),
                axis=-1)
            new_val_x_set[i] = x

        new_test_x_set = np.zeros((testing_x_set.shape[0], testing_x_set.shape[1], testing_x_set.shape[2], new_n_f))
        for i, x in enumerate(testing_x_set):
            record_key = x[0, 0, -2]
            record_key_yesterday = record_key - 24 * 12
            record_key_yesterday = record_key_yesterday if record_key_yesterday >= 0 else record_key + points_per_week - 24 * 12
            x = np.concatenate(
                (x[:, :, 0:1], x[:, :, 2:3], records_time_idx[record_key], records_time_idx[record_key_yesterday]),
                axis=-1)
            new_test_x_set[i] = x

        # Add tailing target values form x values to facilitate local trend attention in decoder
        training_y_set = np.concatenate(
            (new_train_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], training_y_set[:, :, :, :]), axis=1)
        validation_y_set = np.concatenate(
            (new_val_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], validation_y_set[:, :, :, :]), axis=1)
        testing_y_set = np.concatenate(
            (new_test_x_set[:, -1 * self.dec_seq_offset:, :, 0:1], testing_y_set[:, :, :, :]), axis=1)

        # max-min normalization on input and target values
        (stats_x, x_train, x_val, x_test) = z_score_normalize(new_train_x_set, new_val_x_set, new_test_x_set)
        (stats_y, y_train, y_val, y_test) = z_score_normalize(training_y_set, validation_y_set, testing_y_set)

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
        self.dataset = Dataset(data=data, y=y, stats_x=stats_x, stats_y=stats_y)

    def load_edge_data_file(self, filename: str, scaling: bool = True):
        try:
            w = pd.read_csv(filename, header=None).values

            dst_edges = []
            src_edges = []
            edge_attr = []
            for row in range(w.shape[0]):
                for col in range(w.shape[1]):
                    if w[row][col] != 0:
                        dst_edges.append(col)
                        src_edges.append(row)
                        edge_attr.append([w[row][col]])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(edge_attr, scaling)

            self.edge_index = edge_index
            self.edge_attr = edge_attr

        except FileNotFoundError:
            print(f'ERROR: input file was not found in {filename}.')

    def load_semantic_edge_data_file(self, semantic_filename: str, edge_weight_file: str, scaling: bool = True):
        try:
            w = pd.read_csv(edge_weight_file, header=None).values

            semantic_file = open(semantic_filename, 'rb')
            sensor_details = pickle.load(semantic_file)

            dst_edges = []
            src_edges = []
            edge_attr = []
            for i, (sensor, neighbours) in enumerate(sensor_details.items()):
                for src in neighbours:
                    if w[sensor][src] != 0:
                        dst_edges.append(sensor)
                        src_edges.append(src)
                        edge_attr.append([w[sensor][src]])

            edge_index = [src_edges, dst_edges]
            edge_attr = scale_weights(edge_attr, scaling)

            self.edge_index_semantic = edge_index
            self.edge_attr_semantic = edge_attr

        except FileNotFoundError:
            print(f'ERROR: input files was not found')

    def _create_graph(self, x, edge_index, edge_attr):
        graph = data.Data(x=Tensor(x),
                          edge_index=torch.LongTensor(edge_index),
                          y=None,
                          edge_attr=Tensor(edge_attr))
        return graph

    def load_batch(self, _type: str, offset: int, batch_size: int = 32, device: str = 'cpu') -> (list, list):
        to = ToDevice(device)

        xs = self.dataset.get_data(_type)
        ys = self.dataset.get_y(_type)
        limit = (offset + batch_size) if (offset + batch_size) <= len(xs) else len(xs)

        xs = xs[offset: limit, :, :, :]  # [9358, 13, 228, 1]
        ys = ys[offset: limit, :]

        ys_input = np.copy(ys)
        if _type != 'train':
            ys_input[:, self.dec_seq_offset:, :, :] = 0

        feature_xs_graphs = [[] for i in range(self.enc_features)]
        feature_xs_graphs_semantic = [[] for i in range(self.enc_features)]
        num_inner_f_enc = int(xs.shape[-1] / self.enc_features)
        for k in range(self.enc_features):
            batched_xs_graphs = [[] for i in range(batch_size)]
            batched_xs_graphs_semantic = [[] for i in range(batch_size)]

            for idx, x_timesteps in enumerate(xs):
                graph_xs = []
                graph_xs_semantic = []

                for inner_f in range(num_inner_f_enc):
                    start_idx = (k * num_inner_f_enc) + num_inner_f_enc - inner_f - 1
                    end_idx = start_idx + 1
                    [graph_xs.append(to(self._create_graph(x[:, start_idx: end_idx],
                                                           self.edge_index, self.edge_attr))) for x in x_timesteps]
                    [graph_xs_semantic.append(to(self._create_graph(x[:, start_idx: end_idx],
                                                                    self.edge_index_semantic, self.edge_attr_semantic)))
                     for x in x_timesteps]
                # else:
                #     # TODO: This is hard coded. Please replace with a proper index selection
                #     # [graph_xs.append(to(self._create_graph(x[:, 2:3], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last week
                #     # [graph_xs_semantic.append(to(self._create_graph(x[:, 2:3], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last week
                #     [graph_xs.append(to(self._create_graph(x[:, 1:2], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last day
                #     [graph_xs_semantic.append(to(self._create_graph(x[:, 1:2], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last day
                #     [graph_xs.append(to(self._create_graph(x[:, 0:1], self.edge_index, self.edge_attr))) for x in x_timesteps]  # last hour
                #     [graph_xs_semantic.append(to(self._create_graph(x[:, 0:1], self.edge_index_semantic, self.edge_attr_semantic))) for x in x_timesteps]  # last day

                batched_xs_graphs[idx] = graph_xs
                batched_xs_graphs_semantic[idx] = graph_xs_semantic

            feature_xs_graphs[k] = batched_xs_graphs
            feature_xs_graphs_semantic[k] = batched_xs_graphs_semantic

        feature_xs_graphs_all = [feature_xs_graphs, feature_xs_graphs_semantic]

        batched_xs = [[] for i in range(batch_size)]
        for idx, x_timesteps in enumerate(xs):
            speed_vals = np.concatenate(np.array([xs[idx][:, :, 1:2], xs[idx][:, :, 0:1]]), axis=0)
            rep_vals = np.concatenate(np.array([xs[idx][:, :, 3:4], xs[idx][:, :, 2:3]]), axis=0)
            if self.enc_features > 1:
                batched_xs[idx] = torch.Tensor(np.concatenate((speed_vals, rep_vals), axis=-1)).to(device)
            else:
                batched_xs[idx] = torch.Tensor(speed_vals).to(device)
        batched_xs = torch.stack(batched_xs)

        batched_ys = [[] for i in range(batch_size)]  # decoder input
        batched_ys_graphs = [[] for i in range(batch_size)]  # This is for the decoder input graph
        batched_ys_graphs_semantic = [[] for i in range(batch_size)]  # This is for the decoder input graph
        batch_ys_target = [[] for i in range(batch_size)]
        for idx, y_timesteps in enumerate(ys_input):
            graphs_ys = []
            graphs_ys_semantic = []
            for i, y in enumerate(y_timesteps):
                graph = self._create_graph(y, self.edge_index, self.edge_attr)
                graph_semantic = self._create_graph(y, self.edge_index_semantic, self.edge_attr_semantic)
                graphs_ys.append(to(graph))
                graphs_ys_semantic.append(to(graph_semantic))

            batched_ys[idx] = torch.Tensor(ys_input[idx]).to(device)
            batch_ys_target[idx] = torch.Tensor(ys[idx]).to(device)
            batched_ys_graphs[idx] = graphs_ys
            batched_ys_graphs_semantic[idx] = graphs_ys_semantic

        graph_ys_all = [batched_ys_graphs, batched_ys_graphs_semantic]

        batched_ys = torch.stack(batched_ys)

        if not self.graph_enc_input:
            feature_xs_graphs_all = None
        if not self.non_graph_enc_input:
            batched_xs = None
        if not self.graph_dec_input:
            graph_ys_all = None
        if not self.non_graph_dec_input:
            batched_ys = None

        return batched_xs, feature_xs_graphs_all, batched_ys, graph_ys_all, batch_ys_target
