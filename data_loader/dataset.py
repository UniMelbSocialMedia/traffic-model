class Dataset(object):
    def __init__(self, data, y, stats_train, stats_val, stats_test, n_batch_train, n_batch_test, n_batch_val):
        self.__data = data
        self.__y = y
        self.stats_train = stats_train
        self.stats_val = stats_val
        self.stats_test = stats_test
        self.n_batch_train = n_batch_train
        self.n_batch_test = n_batch_test
        self.n_batch_val = n_batch_val

    def get_data(self, _type):
        return self.__data[_type]

    def get_y(self, _type):
        return self.__y[_type]

    def get_len(self, type):
        return len(self.__data[type])

    def get_n_batch_train(self):
        return self.n_batch_train

    def get_n_batch_test(self):
        return self.n_batch_test

    def get_n_batch_val(self):
        return self.n_batch_val

