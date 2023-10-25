import numpy as np
import pickle

from utils.math_utils import normalize


def seq_gen_v2(len_seq, data_seq, n_frame, n_route, C_0=1):
    seq_dataset_size = len_seq - n_frame
    tmp_seq = np.zeros((seq_dataset_size, n_frame, n_route, C_0))
    for i in range(seq_dataset_size):
        start = i
        end = start + n_frame
        tmp_seq[i, :, :, :] = np.reshape(data_seq[start:end, :], [n_frame, n_route, C_0])
    return tmp_seq


def attach_prev_dys_seq(seq: np.array, n_his: int, day_slots: int, num_days_per_week: int, last_week: bool,
                        last_day: bool, total_drop: int):
    seq_tmp = seq[total_drop:]

    seq_output = []
    for k in range(len(seq_tmp)):
        lst_dy_data = seq[total_drop + k - day_slots][:n_his]
        lst_wk_data = seq[total_drop + k - (day_slots * num_days_per_week)][:n_his]

        tmp = seq_tmp[k][:n_his]
        if last_day:
            tmp = np.concatenate((tmp, lst_dy_data), axis=-1)
        if last_week:
            tmp = np.concatenate((tmp, lst_wk_data), axis=-1)

        seq_output.append(tmp)

    return np.array(seq_output)


def derive_rep_timeline(x_set: np.array, points_per_week: int, num_of_vertices: int, load_file=True, output_filename= None):
    """
    For every data point per week, we derive a representation time series.

    Parameters
    ----------
    x_set: training set
    points_per_week: number of data points per week
    num_of_vertices: number of nodes in the road network

    Returns
    -------
    records_time_idx: set of representation vectors per each data point in a week
    """

    if load_file:
        output_file = open(output_filename, 'rb')
        records_time_idx = pickle.load(output_file)
        return records_time_idx

    training_size = x_set.shape[0]
    num_weeks_training = int(training_size / points_per_week)
    seq_len = x_set.shape[1]

    records_time_idx = {}

    for time_idx in range(points_per_week):
        # x and y values representation vectors
        record = [x_set[time_idx]]

        record_key = record[0][0, 0, 1]
        for week in range(1, num_weeks_training + 1):
            idx = time_idx + points_per_week * week
            if idx >= training_size: continue

            record.append(x_set[idx])

        sensor_means = []
        for sensor in range(num_of_vertices):
            sensor_data = np.array(record)[:, :, sensor, 0]
            n_samples = sensor_data.shape[0]

            mean_ts = []
            for t in range(seq_len):
                sensor_t = sensor_data[:, t]

                less_ten = (sensor_t < 10).sum()
                if n_samples == less_ten:
                    mean_ts.append(np.mean(sensor_data[:, t]))
                else:
                    non_zero_idx = list(np.nonzero(sensor_t)[0])
                    mean_ts.append(np.mean(sensor_data[non_zero_idx, t]))

            mean = np.expand_dims(np.array(mean_ts), axis=-1)
            sensor_means.append(mean)

        records_time_idx[record_key] = np.array(sensor_means).transpose(1, 0, 2)

    if output_filename is not None:
        with open(output_filename, 'wb') as file:
            pickle.dump(records_time_idx, file)

    return records_time_idx


def scale_weights(w, scaling=True, min_max=False):
    '''
    Load weight matrix function.
    :param w: list of edge weights
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray.
    '''

    # check whether W is a 0/1 matrix.
    if set(np.unique(w)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling and not min_max:
        return np.log10(w)
    elif scaling and min_max:
        w = np.array(w)
        _min = w.min(axis=(0), keepdims=True)
        _max = w.max(axis=(0), keepdims=True)
        return normalize(w, _max, _min)
    else:
        return w


def search_index(max_len, num_of_depend=1, num_for_predict=12, points_per_hour=12, units=1, offset=0):
    '''
    Parameters
    ----------
    max_len: int, length of all encoder input
    num_of_depend: int,
    num_for_predict: int, the number of points will be predicted for each sample
    units: int, week: 7 * 24, day: 24, recent(hour): 1
    points_per_hour: int, number of points per hour, depends on data
    offset: int, used to get target indexes
    Returns
    ----------
    list[(start_idx, end_idx)]
    '''
    x_idx = []
    for i in range(1, num_of_depend + 1):
        start_idx = max_len - points_per_hour * units * i + offset
        for j in range(num_for_predict):
            end_idx = start_idx + j
            x_idx.append(end_idx)
    return x_idx


def create_lookup_index(last_week=True, last_dy=True, dec_seq_offset=0, dec_seq_len=12):
    wk_lookup_idx = search_index(max_len=0,
                                 units=24 * 7,
                                 offset=0)
    dy_lookup_idx = search_index(max_len=0,
                                 units=24,
                                 offset=0)
    hr_lookup_idx = search_index(max_len=0,
                                 units=1)

    max_lookup_len_enc = min(hr_lookup_idx) * -1
    if last_dy:
        max_lookup_len_enc = min((dy_lookup_idx, hr_lookup_idx))[0] * -1
    if last_week:
        max_lookup_len_enc = min((wk_lookup_idx, dy_lookup_idx, hr_lookup_idx))[0] * -1

    wk_lookup_idx = [x + max_lookup_len_enc for x in wk_lookup_idx]
    dy_lookup_idx = [x + max_lookup_len_enc for x in dy_lookup_idx]
    hr_lookup_idx = [x + max_lookup_len_enc for x in hr_lookup_idx]

    lookup_idx_enc = hr_lookup_idx
    if last_week and last_dy:
        lookup_idx_enc = (wk_lookup_idx + dy_lookup_idx + hr_lookup_idx)
    if not last_week and last_dy:
        lookup_idx_enc = (dy_lookup_idx + hr_lookup_idx)
    if last_week and not last_dy:
        lookup_idx_enc = (wk_lookup_idx + hr_lookup_idx)

    start_idx_lk_dec = max_lookup_len_enc - dec_seq_offset
    lookup_idx_dec = [i for i in range(start_idx_lk_dec, start_idx_lk_dec + dec_seq_len)]
    max_lookup_len_dec = start_idx_lk_dec + dec_seq_len

    return max_lookup_len_enc, lookup_idx_enc, max_lookup_len_dec, lookup_idx_dec
