import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def visualize_confusion_matrix(images, true, predicted, out_path):
    """
    :param images: images used as labels in confusion matrix
    :param true: true labels
    :param predicted: predicted labels
    :param out_path: path for saving the figure
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    cm = metrics.confusion_matrix(true, predicted)
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
