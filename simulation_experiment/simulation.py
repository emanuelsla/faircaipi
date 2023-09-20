import numpy
import numpy as np

from algorithm.biased_decision import is_relevant, is_in_DN_or_FP, is_biased
from algorithm.interaction import correct_explanation, correct_prediction


def to_counter_examples(shap_values: list, x_query: numpy.ndarray, x_idx: numpy.ndarray, protected_idx: int,
                        y_hat: numpy.ndarray, y_U: numpy.ndarray, unfavorable_label: int,
                        min_perc: float) -> (numpy.ndarray, numpy.ndarray):
    """

    Handles label and explanation corrections in simulation mode according to the three cases WWR, RWR and RRR and
    criteria for (un)fair explanations. If a label is corrected  (WWR), no explanation correction is performed. If no
    label correction is performed, an explanation correction is performed only if the explanation is unfair. For an
    explanation to be unfair the protected attribute must be relevant in the explanation and the discriminating groups
    DN or FP must be present. If an explanation correct is performed the case RWR is present. If no explanation
    correction is performed, case RRR is present.

    :param shap_values: list of SHAP values
    :type shap_values: list
    :param x_query: instance values of query x in unlabeled data pool U
    :type: x_query: numpy.ndarray
    :param x_idx: index of query x in unlabeled data pool U
    :type x_idx: numpy.ndarray
    :param protected_idx: index of protected attribute in dataset
    :type protected_idx: int
    :param y_hat: prediction f(x) of model f
    :type y_hat: numpy.ndarray
    :param y_U: ground truth labels of unlabeled data pool U
    :type y_U: numpy.ndarray
    :param unfavorable_label: either 0 or 1 corresponding to the unfavorable label of dataset
    :type unfavorable_label: int
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :return: tuple of counterexample and corresponding label
    """
    if is_pred_correct(y_hat, y_U, x_idx):  # only if prediction is correct, explanation is corrected
        if is_relevant(shap_values, x_idx, protected_idx, unfavorable_label, min_perc):
            if is_in_DN_or_FP(x_query, x_idx, shap_values, protected_idx, unfavorable_label):  # Right for wrong reasons
                print("RWR DN/FP present → correct_explanation")
                c, c_label = correct_explanation(x_query, protected_idx, y_hat)
            else:
                print("RRR DP/FN present → proceed")
                c, c_label = x_query, y_hat  # Right for right reasons
        else:
            print("RRR s not relevant in z_hat → proceed")
            c, c_label = x_query, y_hat  # Right for right reasons
    else:
        print("WWR → label correction")
        c, c_label = correct_prediction(x_query, y_hat)  # Wrong for wrong reasons
    return c, c_label


def is_pred_correct(y_hat: numpy.ndarray, y_U: numpy.ndarray, x_idx: numpy.ndarray):
    """
    Checks whether prediction is correct

    :param y_hat: model prediction
    :type y_hat: numpy.ndarray
    :param y_U: labels of unseen data U
    :type y_U: numpy.ndarray
    :param x_idx: index of query x
    :type x_idx: numpy.ndarray
    :return: bool
    """
    y = y_U[x_idx]
    print("ground truth, prediction:", y, y_hat)
    if y_hat == y:
        return True
    else:
        return False


def compute_fair_explanation_metrics(shap_values_L: list, pre_shap_values_L: list, shap_values_test: list,
                                     pre_shap_values_test: list, X_L: numpy.ndarray, X_test: numpy.ndarray,
                                     protected_idx: int, unfavorable_label: int) -> (float, float, float, float):
    """
    Computes fair explanation metrics for checking the explanation quality with respect to fairness

    :param shap_values_L: Shap values for L from current interation
    :type shap_values_L: list
    :param pre_shap_values_L: Shap values for L from previous iteration
    :type pre_shap_values_L: list
    :param shap_values_test: Shap values for test set from current interation
    :type shap_values_test: list
    :param pre_shap_values_test: Shap values for test set from previous interation
    :type pre_shap_values_test: list
    :param X_L: features for L
    :type X_L: numpy.ndarray
    :param X_test: features for test set
    :type X_test: numpy.ndarray
    :param protected_idx: index of protected attribute
    :type protected_idx: int
    :param unfavorable_label: unfavorable label of target variable
    :type unfavorable_label: int
    :return: float, float, float, float
    """
    # compute bias mask for all instances in discriminating groups DN or FP
    bias_mask = get_DN_or_FP_mask(shap_values_L, X_L, protected_idx, unfavorable_label)
    # for the mean difference, get shap values for unfavorable label in DN or FP from the previous iteration
    pre_biased_shap_values = get_biased_shap_values(bias_mask[:-1], pre_shap_values_L[unfavorable_label],
                                                    X_L[:-1])
    # for the mean difference, get shap values for unfavorable label in DN or FP from the current iteration
    biased_shap_values_L_without_last_one = get_biased_shap_values(bias_mask[:-1],
                                                                   shap_values_L[unfavorable_label][:-1], X_L[:-1])

    # based on the previous and current iteration, compute mean difference of shap values for the protected attribute
    mean_diff_L = compute_mean_shap_value_difference(protected_idx, pre_biased_shap_values,
                                                     biased_shap_values_L_without_last_one)

    biased_shap_values_L = get_biased_shap_values(bias_mask, shap_values_L[unfavorable_label], X_L)

    # based only on the current iteration, compute mean of shap values for the protected attribute
    mean_L = compute_mean_shap_values(protected_idx, biased_shap_values_L)

    # same procedure for test set
    bias_mask = get_DN_or_FP_mask(shap_values_test, X_test, protected_idx, unfavorable_label)

    pre_biased_shap_values_test = get_biased_shap_values(bias_mask, pre_shap_values_test[unfavorable_label],
                                                         X_test)
    biased_shap_values_test = get_biased_shap_values(bias_mask, shap_values_test[unfavorable_label], X_test)

    mean_diff_test = compute_mean_shap_value_difference(protected_idx, pre_biased_shap_values_test,
                                                        biased_shap_values_test)

    mean_test = compute_mean_shap_values(protected_idx, biased_shap_values_test)

    return mean_diff_L, mean_diff_test, mean_L, mean_test


def compute_mean_shap_value_difference(protected_idx: int, pre_shap_values: list, shap_values: list):
    """
    Computes mean difference of Shap values of protected attribute from previous and current iterations

    :param protected_idx: index of protected attribute
    :type protected_idx: int
    :param pre_shap_values: Shap values from previous iteration
    :type pre_shap_values: list
    :param shap_values: Shap values from current iteration
    :type shap_values: list
    :return: float
    """
    def func(x):
        return x[protected_idx]

    diff = np.array(list(filter(func, shap_values))) - np.array(list(filter(func, pre_shap_values)))
    mean = np.mean(diff)

    return mean


def compute_mean_shap_values(protected_idx: int, shap_values: list):
    """
    Compute mean Shap values of protected attribute for current iteration

    :param protected_idx: index of protected attribute
    :type protected_idx: int
    :param shap_values: Shap values of protected attribute
    :type shap_values: list
    :return: float
    """
    def func(x):
        return x[protected_idx]

    protected_shap_values = np.array(list(filter(func, shap_values)))
    mean = np.mean([np.abs(i) for i in protected_shap_values])
    return mean


def get_DN_or_FP_mask(shap_values: list, X: numpy.ndarray, protected_idx: int, unfavorable_label: int):
    """
    Generate a mask that determines for each instance of a dataset whether it belongs to groups DN or FP

    :param shap_values: Shap values of current iteration
    :type shap_values: list
    :param X: Dataset for which mask is generated
    :type X: numpy.ndarray
    :param protected_idx: index for protected attribute
    :type protected_idx: int
    :param unfavorable_label: unfavorable label of target class
    :type unfavorable_label: int
    :return: DN-FP-mask for dataset
    """
    biased_L = [is_in_DN_or_FP(instance, [idx], shap_values, protected_idx, unfavorable_label) for
                idx, instance in enumerate(X)]
    return biased_L


def get_bias_mask(shap_values: list, X: numpy.ndarray, protected_idx: int, unfavorable_label: int, min_perc: float):
    """
    Generate a mask that determines for each instance of a dataset whether it is biased

    :param shap_values: Shap values of current iteration
    :type shap_values: list
    :param X: Dataset for which mask is generated
    :type X: numpy.ndarray
    :param protected_idx: index for protected attribute
    :type protected_idx: int
    :param unfavorable_label: unfavorable label of target class
    :type unfavorable_label: int
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :return: Bias-mask for dataset
    """

    biased_L = [is_biased(shap_values, instance, [idx], protected_idx, unfavorable_label, min_perc) for
                idx, instance in enumerate(X)]
    return biased_L


def count_explanations(bias_mask):
    count = 0
    for idx, instance in enumerate(bias_mask):
        if bias_mask[idx]:
            count = count + 1
    return count


def get_biased_shap_values(bias_mask: list, shap_values: list, X_L: numpy.ndarray):
    """
    Filter for all biased Shap values

    :param bias_mask: mask that indicates which instances from dataset are biased
    :type bias_mask: list
    :param shap_values: Shap values for the protected attribute from current iteration
    :type shap_values: list
    :param X_L: dataset for which biased Shap values are filtered
    :type X_L: numpy.ndarray
    :return: list of biased shap values
    """
    biased_shap_values = []
    for idx, instance in enumerate(X_L):
        if bias_mask[idx]:
            biased_shap_values.append(list(shap_values[idx]))
    return biased_shap_values
