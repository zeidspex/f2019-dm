import io
import os
import pickle
import tarfile
import keras as ks
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import sklearn.metrics as metrics


class Classifier:
    """
    A classifier based on K-means clustering and previously created
    mappings between cluster labels and true labels
    """
    def __init__(self, embedding_model, kmeans):
        """
        :param embedding_model: Keras model for converting images to embeddings
        :param kmeans: K-means model
        """
        self.embedding_model = embedding_model
        self.kmeans = kmeans

    def predict(self, x):
        """
        :param x: features (images)
        :return: predicted classes
        """
        z = self.embedding_model.predict(x)
        yp = np.array(self.kmeans.predict(z))

        return yp


def cluster_data(z_train, centroids):
    """
    :param z_train: training data embeddings
    :param centroids: initial centroids of the cluster
    :return: predicted labels for training data (yp_train)
    """
    return cluster.KMeans(
        n_clusters=10, max_iter=1, n_jobs=-1, n_init=1, init=centroids
    ).fit(z_train)


def create_model(autoencoder, embedding_layer, x_train, y_train, sample_size):
    """
    :param autoencoder: trained autoencoder model
    :param embedding_layer: index of embedding layer
    :param x_train: training features
    :param y_train: training labels
    :param sample_size: sample size for cluster labeling
    :return: a classifier
    """
    # Create embedding model
    embedding_model = ks.models.Model(
        inputs=autoencoder.inputs, outputs=autoencoder.layers[embedding_layer].output
    )

    # Train K-means model
    z_train = embedding_model.predict(x_train)
    centroids = np.array([
        np.mean(
            z_train[np.argwhere(y_train == i)].reshape(-1, z_train.shape[1])[0:sample_size],
            axis=0
        )
        for i in range(10)
    ])
    kmeans = cluster_data(z_train, centroids)

    # Create classifier from embeddings model and K-means model and return it
    return Classifier(embedding_model, kmeans)


def load_model(model_path):
    """
    :param model_path: classifier path
    :return: load classifier from hard drive
    """
    with tarfile.open(model_path, mode='r') as in_file:
        with open('embeddings.h5', 'wb') as out_file:
            out_file.write(in_file.extractfile('embeddings.h5').read())

        embeddings = ks.models.load_model('embeddings.h5')
        os.remove('embeddings.h5')
        kmeans = pickle.loads(in_file.extractfile('kmeans.pkl').read())

        return Classifier(embeddings, kmeans)


def save_model(model, model_path):
    """
    :param model: model to save
    :param model_path: output path
    :return: save classifier to the hard drive
    """
    ks.models.save_model(model.embedding_model, model_path)

    with open(model_path, 'rb') as in_file:
        embedding_model = in_file.read()

    with tarfile.open(model_path, mode='w') as out_file:
        names = ['embeddings.h5', 'kmeans.pkl']
        objects = [
            embedding_model,
            pickle.dumps(model.kmeans),
        ]

        for name, data in zip(names, objects):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            out_file.addfile(info, io.BytesIO(data))


def test_model(clf, x_test, y_test, class_names, out_path=None):
    """
    :param clf: classifier to test
    :param x_test: features
    :param y_test: labels
    :param class_names: class names
    :param out_path: path to save the CSV to (including file name)
    :return: None
    """
    yp_test = clf.predict(x_test)
    precision = metrics.precision_score(y_test, yp_test, average=None)
    recall = metrics.recall_score(y_test, yp_test, average=None)
    f1 = metrics.f1_score(y_test, yp_test, average=None)

    pd.set_option('display.max_columns', 10)
    data = np.array(list(zip(precision, recall, f1)))
    data = pd.DataFrame(
        data.T, columns=class_names, index=['Precision', 'Recall', 'F1 Score']).round(2)

    print(data.loc['Precision'].mean())
    print(data.loc['Recall'].mean())

    if out_path:
        data.to_csv(out_path)

    print(data)
