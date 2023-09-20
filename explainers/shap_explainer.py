import aif360.datasets
import numpy
import pandas
import shap
import numpy as np
import pandas as pd
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from modAL import ActiveLearner

from algorithm.interaction import get_label_description


def init_tree_explainer(model: ActiveLearner, X: numpy.ndarray):
    """
    Initializes a TreeExplainer instance on the given dataset

    :param model: ActiveLearner that provides the estimator on which it is based
    :type model: ActiveLearner
    :param X: unlabeled data pool
    :type X: numpy.ndarray
    :return: TreeExplainer initialized on the data
    """

    # see Molnar: https://github.com/christophM/interpretable-ml-book/blob/master/scripts/shap/shap-notebook.ipynb
    explainer = shap.TreeExplainer(model=model.estimator, data=X, model_output="margin",
                                   feature_pertubation="interventional")
    return explainer


def generate_explainer(model: ActiveLearner, X: numpy.ndarray):
    """
    Generates an explainer on the training data X

    :param model: Active Learner SHAP gets predictions from
    :type model: ActiveLearner
    :param X: Dataset for which the learner makes predictions
    :type X: numpy.ndarray
    :return: explainer model and SHAP values
    """

    explainer = init_tree_explainer(model, X)
    shap_values = explainer.shap_values(X, check_additivity=False)
    return explainer, shap_values


def display_explanation(explainer: shap.explainers._tree.Tree, X_U: numpy.ndarray, y_U: numpy.ndarray,
                        U: aif360.datasets.Dataset, prediction, shap_values: list, x_idx: numpy.ndarray,
                        feature_names: pandas.core.indexes.base.Index):
    ground_truth_label = y_U[x_idx]

    if ground_truth_label == prediction:
        accurate = 'Correct'
    else:
        accurate = 'Incorrect'

    unfavorable_label = not U.favorable_label

    shap_values_for_query = shap_values[unfavorable_label][x_idx[0], :]
    feature_importance = pd.DataFrame(list(zip(feature_names, X_U[x_idx, :][0],
                                               ['{:f}'.format(i) for i in shap_values_for_query],
                                               ['{:f}'.format(np.abs(i)) for i in shap_values_for_query])),
                                                columns=['feature_name', 'feature_value', 'shap_value', 'relevance'])
    feature_importance.sort_values(by=['relevance'],
                                   ascending=False, inplace=True)

    print('***' * 35, "\n")
    print(f'Ground Truth Label: {ground_truth_label} {get_label_description(ground_truth_label)} \n')
    print(f'Model Prediction:  {prediction} {get_label_description(prediction)} -- {accurate} \n')

    print(f'Shap values for all attributes sorted by relevance: \n')
    display(feature_importance)
    print()
    print('***' * 35, "\n")

    shap.initjs()
    shap_values_for_query = shap_values[unfavorable_label][x_idx, :]

    # generate local explanation for query x
    explanation = shap.force_plot(explainer.expected_value[1], shap_values_for_query, X_U[x_idx, :],
                                  feature_names=feature_names, matplotlib=True, show=False, figsize=(100, 8),
                                  text_rotation=45, contribution_threshold=0.05)
    plt.tight_layout()
    # plt.savefig("tmp.svg")
    plt.show()
    plt.close()
    return explanation

