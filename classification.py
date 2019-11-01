import io
import os
import json
import pickle
import tarfile
import keras as ks
import numpy as np


class Classifier:
    """
    A classifier based on K-means clustering and previously created
    mappings between cluster labels and true labels
    """
    def __init__(self, embedding_model, kmeans, mappings):
        """
        :param embedding_model: Keras model for converting images to embeddings
        :param kmeans: K-means model
        :param mappings: mappings from cluster labels to true labels
        """
        self.embedding_model = embedding_model
        self.kmeans = kmeans
        self.mappings = mappings

    def predict(self, x):
        """
        :param x: features (images)
        :return: predicted classes
        """
        z = self.embedding_model.predict(x)
        yp = np.array([self.mappings[int(yp)] for yp in self.kmeans.predict(z)])

        return yp


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
        mappings = {
            int(k): v
            for k, v in json.loads(in_file.extractfile('mappings.json').read()).items()
        }

        return Classifier(embeddings, kmeans, mappings)


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
        names = ['embeddings.h5', 'kmeans.pkl', 'mappings.json']
        objects = [
            embedding_model,
            pickle.dumps(model.kmeans),
            json.dumps(model.mappings).encode('UTF-8')
        ]

        for name, data in zip(names, objects):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            out_file.addfile(info, io.BytesIO(data))
