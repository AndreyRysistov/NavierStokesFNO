from tensorflow.keras import layers
from tensorflow.keras import models
from functools import partial
from utils.slice_assing import slice_assign
import tensorflow as tf
import numpy as np


class FConv2D(tf.keras.layers.Layer):

    def __init__(self, output_dim, modes1, modes2, **kwargs):
        self.output_dim = output_dim
        self.modes1 = modes1
        self.modes2 = modes2
        super(FConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weights1 = self.add_weight(shape=(input_shape[-1], self.output_dim, self.modes1, self.modes2, 2),
                                        initializer='random_normal',
                                        trainable=True,
                                        name='w1'
                                        )
        self.weights2 = self.add_weight(shape=(input_shape[-1], self.output_dim, self.modes1, self.modes2, 2),
                                        initializer='random_normal',
                                        trainable=True,
                                        name='w2'
                                        )
        super(FConv2D, self).build(input_shape)

    def compl_mul2d(self, a, b):
        op = partial(tf.einsum, "bctq,dctq->bdtq")
        return tf.stack([
            op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
            op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
        ], axis=-1)

    def call(self, x):
        #batch_size = tf.shape(x)[0]
        #size_x = x.shape[1]
        #size_y = x.shape[2]
        # out_ft = tf.zeros((batch_size, self.out_channels, size_x, size_y//2+1, 2))
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x_ft = tf.signal.rfft2d(x) / (x.shape[-1] * x.shape[-2]) ** 0.5
        real = tf.expand_dims(tf.math.real(x_ft), axis=-1)
        image = tf.expand_dims(tf.math.imag(x_ft), axis=-1)
        x_ft = tf.concat([real, image], axis=-1)
        x_ft = slice_assign(
            x_ft,
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1),
            slice(None, None, None),
            slice(None, None, None),
            slice(None, self.modes1, None),
            slice(None, self.modes2, None),
        )
        x_ft = slice_assign(
            x_ft,
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2),
            slice(None, None, None),
            slice(None, None, None),
            slice(x_ft.shape[2] - self.modes1, None, None),
            slice(None, self.modes2, None),
        )
        output = tf.signal.irfft(tf.complex(x_ft[..., 0], x_ft[..., 1])) * (x.shape[-1] * x.shape[-2]) ** 0.5
        output = tf.transpose(output, perm=[0, 2, 3, 1])

        return output

    def get_config(self):
        config = super(FConv2D, self).get_config()
        config.update({"output_dim": self.output_dim, "modes1": self.modes2, "modes2": self.modes2})
        return config


class BlockFNO2D(layers.Layer):

    def __init__(self, out_channels, modes1, modes2, **kwargs):
        self.output_dim = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.fconv = FConv2D(self.output_dim, self.modes1, self.modes2)
        self.conv = layers.Conv1D(self.output_dim, 1)
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()

        super(BlockFNO2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BlockFNO2D, self).build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        size_x, size_y = x.shape[1], x.shape[2]
        channels = x.shape[3]
        x1 = self.fconv(x)
        x2 = self.conv(tf.reshape(x, (batch_size, size_x * size_y, channels)))
        x2 = tf.reshape(x2, (batch_size, size_x, size_y, channels))
        x = self.bn(x1 + x2)
        x = self.activation(x)
        return x

    def get_config(self):
        config = super(BlockFNO2D, self).get_config()
        config.update({"output_dim": self.output_dim, "modes1": self.modes2, "modes2": self.modes2})
        return config


class FNO2D(models.Model):

    def __init__(self, config,  *args,  **kwargs):
        self.T = config.T
        self.step = config.step
        self.batch_size = config.batch_size

        self.gridx = np.array(np.linspace(0, 1, config.S))
        self.gridx = self.gridx.reshape((1, config.S, 1, 1)).repeat(config.S, axis=2)
        self.gridy = np.array(np.linspace(0, 1, config.S))
        self.gridy = self.gridy.reshape((1, 1, config.S, 1)).repeat(config.S, axis=1)

        inputs = layers.Input(shape=config.model.input_shape)
        x = layers.Dense(config.model.width, activation='linear')(inputs)
        for _ in range(config.model.fno.count):
            x = BlockFNO2D(
            config.model.width,
            config.model.fno.modes1,
            config.model.fno.modes2)(x)

        for j in range(config.model.dense.count):
            x = layers.Dense(config.model.dense.nunits[j], activation=config.model.dense.activation)(x)
        out = layers.Dense(1, activation='linear')(x)
        super(FNO2D, self).__init__(inputs, out)

    def train_step(self, data):
        if len(data) == 3:
            xx, yy, sample_weight = data
        else:
            sample_weight = None
            xx, yy = data

        loss = 0
        with tf.GradientTape() as tape:
            for t in range(0, self.T, self.step):
                y = yy[..., t:t + self.step]
                y_pred = self(xx, training=True)
                loss += self.compiled_loss(
                    y_pred,
                    y,
                    sample_weight=sample_weight,
                    regularization_losses=self.losses,
                )
                if t == 0:
                    pred = y_pred
                else:
                    pred = tf.concat((pred, y_pred), -1)
                xx = tf.concat((
                    xx[..., self.step:-2],
                    y_pred,
                    self.gridx.repeat(self.batch_size, axis=0),
                    self.gridx.repeat(self.batch_size, axis=0)
                ),
                    axis=-1)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        xx, yy = data
        loss = 0
        for t in range(0, self.T, self.step):
            y = yy[..., t:t + self.step]
            y_pred = self(xx, training=False)  # Forward pass
            loss += self.compiled_loss(
                y_pred,
                y,
                regularization_losses=self.losses,
            )
            if t == 0:
                pred = y_pred
            else:
                pred = tf.concat((pred, y_pred), -1)
            xx = tf.concat((
                xx[..., self.step:-2],
                y_pred,
                self.gridx.repeat(self.batch_size, axis=0),
                self.gridy.repeat(self.batch_size, axis=0)
            ),
            axis=-1)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
