import numpy as np
import h5py
from scipy import io
import os

class DataReader():

    def __init__(self, file_path):
        super(DataReader, self).__init__()
        self.data = None
        self.old_mat = None
        self.to_float = True
        self._load_file(file_path)

    def _load_file(self, file_path):
        try:
            self.data = io.loadmat(file_path)
            self.old_mat = True
        except Exception as err:
            self.data = h5py.File(file_path, 'r')
            self.old_mat = False

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[:]
            #x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        return x

    def set_float(self, to_float):
        self.to_float = to_float