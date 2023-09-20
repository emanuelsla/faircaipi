import numpy as np


def get_contribution_threshold(shap_values: list, x_idx: np.ndarray, unfavorable_label: int, min_perc: float) -> float:
    """
    Computes the contribution threshold based on the minimum percentage (default value is according to Lundberg et al.
    0.05 corresponding to 5 percent) to which the Shap values of all attributes are compared in order to examine which
    attributes contribute at least minimum percentage to the prediction.
    Official description of contribution threshold in force plots: "Controls the feature names/values that are displayed
    on force plot. Only features that the magnitude of their shap value is larger than
    min_perc * (sum of all abs shap values) will be displayed." (Lundberg et al.)
    see: https://github.com/slundberg/shap/blob/b6e90c859fdfc6bc145242d9a8082d4ad844e995/shap/plots/_force.py

    :param shap_values: list of Shap values
    :type shap_values: list
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: np.ndarray
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :return: float contribution threshold to which the Shap values are compared
    """
    sum = 0
    for val in shap_values[unfavorable_label][x_idx[0], :]:
        sum = sum + np.abs(val)

    for val in shap_values[not unfavorable_label][x_idx[0], :]:
        sum = sum + np.abs(val)
    # contribution_threshold = 0.05*(sum of all abs shap_values)
    contribution_threshold = min_perc * sum
    return contribution_threshold


def is_relevant(shap_values: list, x_idx: np.ndarray, protected_idx: int, unfavorable_label: int,
                min_perc: float) -> bool:
    """
    Checks whether the Shap value of protected attribute s is relevant in the explanation according to the
    contribution threshold of the force plot.

    :param shap_values: list of Shap values
    :type shap_values: list
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: np.ndarray
    :param protected_idx: index of protected attribute in dataset
    :type protected_idx: int
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :return: bool corresponding to relevant (True) or irrelevant (False)
    """
    contribution_threshold = get_contribution_threshold(shap_values, x_idx, unfavorable_label, min_perc)
    shap_values_for_query = shap_values[unfavorable_label][x_idx[0], :]
    if np.abs(float(shap_values_for_query[protected_idx])) > contribution_threshold:
        return True
    return False


def is_privileged(x_query: np.ndarray, protected_idx: int) -> bool:
    """
    Checks whether query is privileged or unprivileged

    :param x_query: instance values of query x in unlabeled data pool U
    :type x_query: np.ndarray
    :param protected_idx: index of protected attribute in dataset
    :type protected_idx: int
    :return: bool corresponding to privileged (True) or unprivileged (False)
    """

    # 1 corresponding to the privileged attribute value
    if x_query[protected_idx] == 1:
        return True
    else:
        return False


def increases_unfavorable_class_probability(shap_values: list, x_idx: np.ndarray, protected_idx: int,
                                            unfavorable_label: int) -> bool:
    """
    Checks whether shap value of protected attribute increases (positive Shap value) or decreases (negative Shap value)
    the unfavorable class probability. When considering the explanation for the unfavorable class,
    positive shap values indicate that the probability for the unfavorable class increases.
    If s does not have a negative sign, it contributes to increasing the probability for the unfavorable class of
    receiving a bad prediction.

    :param shap_values: list of SHAP values
    :type shap_values: list
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: np.ndarray
    :param protected_idx: index of protected attribute in dataset
    :type protected_idx: int
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :return: bool corresponding to increases (True) or decreases (False)
    """
    if shap_values[unfavorable_label][x_idx[0], :][protected_idx] > 0:
        return True
    else:
        return False


def is_biased(shap_values: list, x_query: np.ndarray, x_idx: np.ndarray, protected_idx: int,
              unfavorable_label: int, min_perc: float) -> bool:
    """
    Checks whether a prediction is biased with respect to relevance of protected attribute and group membership tp DN or
    FP

    :param shap_values: list of SHAP values
    :type shap_values: list
    :param x_query: instance values of query x in unlabeled data pool U
    :type x_query: np.ndarray
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: np.ndarray
    :param protected_idx: index of protected attribute in dataset
    :type protected_idx: int
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :return: bool
    """
    if is_relevant(shap_values, x_idx, protected_idx, unfavorable_label, min_perc):
        if is_in_DN_or_FP(x_query, x_idx, shap_values, protected_idx, unfavorable_label):
            print("FP or DN is present, therefore decision is biased")
            return True
        else:
            print("FN or DP is not present, therefore decision is not biased")
            return False
    else:
        print("Protected attribute is not relevant in the explanation")
        return False


def is_in_DN_or_FP(x_query: np.ndarray, x_idx: np.ndarray, shap_values: list, protected_idx: int,
                   unfavorable_label: int) -> bool:
    """
    Checks whether query belongs to discriminating groups DN (unprivileged and predicted unfavorable label) or FP
    (privileged and predicted favorable label) or otherwise to non-discriminating groups DP
    (unprivileged and predicted favorable label) or FN (privileged and predicted unfavorable label)

    :param x_query: instance values of query x in unlabeled data pool U
    :type x_query: np.ndarray
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: np.ndarray
    :param shap_values: list of SHAP values
    :type shap_values: list
    :param protected_idx:  index of protected attribute in dataset
    :type protected_idx: int
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :return: bool corresponding to is in DN or FP (True) or is not in DN or FP (False)
    """

    if is_privileged(x_query, protected_idx) == True and increases_unfavorable_class_probability(
            shap_values, x_idx, protected_idx, unfavorable_label) == False:
        # print("FP is present")
        return True
    elif is_privileged(x_query, protected_idx) == False and increases_unfavorable_class_probability(
            shap_values, x_idx, protected_idx, unfavorable_label) == True:
        # print("DN is present")
        return True
    else:
        # print("Since DN or FP is not present, FN or DP is present")
        return False
