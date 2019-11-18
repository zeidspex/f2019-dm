import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def visualize_confusion_matrix(x_test, y_test, yp_test, out_path):
    """
    :param x_test: features (needed for visualizing classes in confusion matrix)
    :param y_test: labels
    :param yp_test: predicted labels
    :param out_path: path for saving the figure
    :return: None
    """
    images = [
        (x_test[np.argwhere(y_test == i)][0] * 255).astype('uint8').reshape((x_test.shape[1:]))
        for i in range(10)
    ]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    cm = metrics.confusion_matrix(y_test, yp_test)
    ax.matshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center", color="white", size=24
            )

    for i, y in enumerate(range(10)):
        y = 10.5 + 8 * i

        for xy, idx in zip((lambda x: [x, x[::-1]])([0.915, y / 100]), [9 - i, i]):
            ax1 = fig.add_axes([*xy, 0.07, 0.07])
            ax1.axison = False
            ax1.imshow(images[idx])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Predicted', size=24)
    ax.set_ylabel('True', size=24)

    plt.savefig(out_path)
    plt.show()


def plot_loss(history_path, out_path):
    """
    :param history_path: path to training history file
    :param out_path: path for saving the figure
    :return: None
    """
    with open(history_path, 'rb') as file:
        data = pickle.loads(file.read())

    plt.style.use("seaborn")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data['loss'], label='Training Loss')
    ax.plot(data['val_loss'], label='Validation Loss')
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)

    plt.savefig(out_path)
    plt.show()
