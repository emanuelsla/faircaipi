import aif360.datasets

import numpy
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from IPython.core.display_functions import display
from sklearn.base import BaseEstimator


def fit(estimator: BaseEstimator, X_L: numpy.ndarray,
        y_L: numpy.ndarray) -> ActiveLearner:

    """
    Initializes ActiveLearner object with the given training set and classifier

    :param estimator: sklearn base estimator that functions as the underlying model of the ActiveLearner
    :type estimator: BaseEstimator
    :param X_L: training set without labels on which the estimator is trained on
    :type X_L: numpy.ndarray
    :param y_L: labels for the training set
    :type y_L: numpy.ndarray
    :return: ActiveLearner
    """
    learner: ActiveLearner = ActiveLearner(
        estimator=estimator,
        query_strategy=uncertainty_sampling,
        X_training=X_L, y_training=y_L)
    return learner


def select_query(model: ActiveLearner, X_U: numpy.ndarray):
    """
    Selects a query using the sampling strategy specified in initialization of ActiveLearner model

    :param model: learner that selects a query
    :type model: ActiveLearner
    :param X_U: unlabeled data pool without labels y from which the learner selects a query
    :type X_U: numpy.ndarray
    :return: index and query instance of selected query
    """

    # select_query(f,U)
    x_idx, x_query = model.query(X_U)
    return x_idx, x_query[0]


def present_query(U: aif360.datasets.Dataset, x_idx: numpy.ndarray):
    print('***'*35, "\n")
    print("Selected query x at index:", x_idx, "\n")
    display(U.convert_to_dataframe()[0].iloc[x_idx[0]])
    print('***'*35, "\n")
