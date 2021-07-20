import numpy as np
from dataloaders.data_reader import DataReader


class DataLoader(object):

    def __init__(self, config):
        super(DataLoader, self).__init__()
        self.config = config
        self.reader = DataReader(self.config.data_path)

    def load_data(self, subset='training'):
        examples_slice = slice(None, self.config.ntrain)
        if subset == 'test':
            examples_slice = slice(-self.config.ntest)
        data_a = self.reader.read_field('u')[
                  examples_slice,
                  ::self.config.sub,
                  ::self.config.sub,
                  :self.config.T_in
                  ]
        data_u = self.reader.read_field('u')[
                  examples_slice,
                  ::self.config.sub,
                  ::self.config.sub,
                  self.config.T_in:self.config.T + self.config.T_in]
        if self.config.mode == '2D':
            data_a = self.set_grid2d(data_a)
        print('Train shapes: a:{}, u:{}'.format(data_a.shape, data_u.shape))
        return data_a, data_u

    def set_grid2d(self, data):
        gridx = np.array(np.linspace(0, 1, self.config.S))
        gridx = gridx.reshape((1, self.config.S, 1, 1)).repeat(self.config.S, axis=2)

        gridy = np.array(np.linspace(0, 1, self.config.S))
        gridy = gridy.reshape((1, 1, self.config.S, 1)).repeat(self.config.S, axis=1)
        n = data.shape[0]
        data = np.concatenate([
            data,
            gridx.repeat(n, axis=0),
            gridy.repeat(n, axis=0),
        ], axis=-1)
        return data