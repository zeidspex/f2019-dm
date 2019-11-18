import numpy as np
import pandas as pd
import sklearn.metrics as metrics


def test_classifier(clf, x_test, y_test, class_names, out_path=None):
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
