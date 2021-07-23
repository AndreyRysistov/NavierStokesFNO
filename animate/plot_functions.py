import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML
from celluloid import Camera
import numpy as np

matplotlib.use('Agg')


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Model loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Test'])

    axes[1].plot(history.history['mae'])
    axes[1].plot(history.history['val_mae'])
    axes[1].set_title('Model MAE')
    axes[1].set_ylabel('MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Test'])
    plt.show()
    return fig


def animate_prediction(model, datagen, step=1, T=10):
    fig, axes = plt.subplots (1, 2, figsize=(10, 8))
    camera = Camera(fig)
    for xx, yy in zip(datagen[0][0], datagen[0][1]):
        x = np.expand_dims(xx, axis=0)
        for t in range(0, T):
            y = yy[..., t:t + step]
            y_pred = model.predict(x)
            x = np.concatenate((
                x[..., step:-2],
                y_pred,
                model.gridx.repeat(1, axis=0),
                model.gridy.repeat(1, axis=0)
            ),
            axis=-1)
            axes[0].imshow(y);
            axes[1].imshow(y_pred.reshape(y_pred.shape[-3:]));
            axes[0].set_title('True behavior')
            axes[0].set_title('Predict behavior')
            camera.snap()
    animation = camera.animate(interval=500, blit=True)
    HTML(animation.to_html5_video())
    return animation
