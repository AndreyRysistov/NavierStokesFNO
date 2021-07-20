from math import ceil
import numpy as np
from tensorflow.keras import utils


class DataGenerator(utils.Sequence):

    def __init__(self, config, dataloader, subset='training', shuffle=False):
        self.x, self.y = dataloader.load_data(subset=subset)
        self.batch_size = config.batch_size
        if shuffle:
            s = np.arange(self.x.shape[0])
            s = np.random.shuffle(s)
            self.x = self.x[s].reshape(*self.x.shape)
            self.y = self.y[s].reshape(*self.y.shape)

    def __len__(self):
        return ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        end = min(self.x.shape[0], (idx + 1) * self.batch_size)
        return self.x[idx * self.batch_size:end], self.y[idx * self.batch_size:end]