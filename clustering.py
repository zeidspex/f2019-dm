import numpy as np
import sklearn.cluster as cluster


def cluster_data(z_train):
    """
    :param z_train: training data embeddings
    :return: predicted labels for training data (yp_train)
    """
    return cluster.KMeans(n_clusters=10, max_iter=1000000, n_jobs=-1, n_init=10).fit(z_train)


def create_samples(y_train, yp_train, n):
    """
    :param y_train: training data labels
    :param yp_train: predicted training data labels (clusters)
    :param n: number of samples from each cluster
    :return:
        A list of tuples, with each tuple consisting of a cluster label
        and an array of true labels - samples which were put in this cluster
    """
    return list(enumerate([
        y_train[np.random.choice(np.argwhere(yp_train == i).reshape(-1), n)]
        for i in range(10)
    ]))


def map_clusters(labels, mmap):
    """
    :param labels: an array of tuples, returned from create_sample function
    :param mmap:
        map one true class to multiple clusters.
        Higher accuracy for some classes, but other classes are not recognized at all
    :return: a dictionary which contains mappings of cluster labels to true labels
    """
    mappings = {}

    while labels:
        labels.sort(key=lambda x: np.max(np.bincount(x[1])))
        true, predicted = labels.pop()
        bins = np.bincount(predicted, minlength=11)

        if not mmap:
            for v in mappings.values():
                bins[0] = -1
                bins[v] = -1

        mappings[int(true)] = int(np.argmax(bins))

    return mappings
