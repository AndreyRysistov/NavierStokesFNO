import os
from tensorflow.keras import callbacks
from trainers.trainer_setup import *
from animate.plot_functions import plot_history, animate_prediction


class ModelTrainer:

    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.callbacks = []
        self._init_callbacks()

    def train(self, train_data, val_data):
        history = self._fit(train_data, val_data)
        self.model.load_weights(os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'))
        self.model.save(os.path.join(self.config.callbacks.checkpoint.dir, 'model.hdf5'))
        self._save_history(history=history)
        self._animate_prediction(data=train_data)
        print('Model training completed successfully')
        return self.model

    def test(self, data):
        self.model.load_weights(os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'))
        scores = self.model.evaluate(data, verbose=1)
        self._animate_prediction(data=data)
        print("Achieved an MAE: %.2f%%\n" % (scores[1]))

    def _animate_prediction(self, data):
        animation = animate_prediction(self.model, data)
        path = os.path.join(self.config.graphics.dir, 'navie_stokes.mp4')
        animation.save(path, dpi=100, savefig_kwargs={'frameon': False, 'pad_inches': 0})
        print(f'Animation of predictions was saved to {path}')

    def _save_history(self, history, step=0):
        path = os.path.join(self.config.graphics.dir, 'history')
        plot_history(history).savefig(path)
        print(f'Graph of history of the loss function and accuracy was saved to {path}')

    def _fit(self, train_data, val_data, step=0):
        optimizer_name = self.config.trainer.optimizer.name.lower()
        optimizer_params = self.config.trainer.optimizer.params.toDict()
        optimizer = optimizers[optimizer_name](**optimizer_params)
        loss_name = self.config.trainer.loss.name.lower()
        loss_function = losses[loss_name]
        metrics = self.config.trainer.metrics
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=metrics
        )
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.trainer.num_epochs,
            batch_size=self.config.batch_size,
            callbacks=self.callbacks,
        )
        return history

    def _init_callbacks(self):
        if self.config.callbacks.checkpoint.exist:
            self.callbacks.append(
                callbacks.ModelCheckpoint(
                    filepath=os.path.join(self.config.callbacks.checkpoint.dir, 'best_model.hdf5'),
                    monitor=self.config.callbacks.checkpoint.monitor,
                    mode=self.config.callbacks.checkpoint.mode,
                    save_best_only=self.config.callbacks.checkpoint.save_best_only,
                    save_weights_only=self.config.callbacks.checkpoint.save_weights_only,
                    verbose=self.config.callbacks.checkpoint.verbose,
                )
            )
        if self.config.callbacks.tensor_board.exist:
            self.callbacks.append(
                callbacks.TensorBoard(
                    log_dir=self.config.callbacks.tensor_board.log_dir,
                    write_graph=self.config.callbacks.tensor_board.write_graph,
                )
            )
        if self.config.callbacks.early_stopping.exist:
            self.callbacks.append(
                callbacks.EarlyStopping(
                    monitor=self.config.callbacks.early_stopping.monitor,
                    patience=self.config.callbacks.early_stopping.patience,
                    restore_best_weights=self.config.callbacks.early_stopping.restore_best_weights
                )
            )
        if self.config.callbacks.cyclic_lr.exist:
            self.callbacks.append(
                callbacks.CyclicLR(
                    base_lr=self.config.callbacks.cyclic_lr.base_lr,
                    max_lr=self.config.callbacks.cyclic_lr.max_lr,
                    step_size=self.config.callbacks.cyclic_lr.step_size,
                    mode=self.config.callbacks.cyclic_lr.mode,
                    gamma=self.config.callbacks.cyclic_lr.gamma
                )
            )
        if self.config.callbacks.reduce_lr_on_plateau.exist:
            self.callbacks.append(
                callbacks.ReduceLROnPlateau(
                    monitor=self.config.callbacks.reduce_lr_on_plateau.monitor,
                    factor=self.config.callbacks.reduce_lr_on_plateau.factor,
                    patience=self.config.callbacks.reduce_lr_on_plateau.patience,
                    min_lr=self.config.callbacks.reduce_lr_on_plateau.min_lr
                )
            )
