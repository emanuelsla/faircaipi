import aif360.datasets
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from modAL import ActiveLearner


def compute_fairness_metrics(model: ActiveLearner, test: aif360.datasets.Dataset, L: aif360.datasets.Dataset, s: str):
    """
    Computes fairness metrics from AIF360 for dataset and model predictions on L and test set. Statistical Parity is
    computed on L since it changes in each iteration and is expected to become fairer with more counterexamples. The
    test set is chosen for evaluating the fairness of the model using all other fairness metrics which are based on
    the model's predictions on unseen data.

    :param model: predicts the test set labels on which the metrics are computed
    :type model: ActiveLearner
    :param test: separate test set on which the model was not trained on
    :type test: aif360.datasets.Dataset
    :param L: labeled data pool in AIF360 StandardDataset format on which the model is trained on
    :type test: aif360.datasets.Dataset
    :param s: selected binary protected attribute, e.g. 'sex', 'age', 'race' or other
    :type s: str
    :return: float values corresponding to fairness metrics
    """

    # Get privileged and unprivileged attribute values
    privileged_protected_value = L.privileged_protected_attributes[0][0]
    unprivileged_group = [{s: int(not privileged_protected_value)}]
    privileged_group = [{s: privileged_protected_value}]

    # Get predictions from f
    dataset_test_preds_before = model.predict(test.features)
    predictions_before = test.copy()
    predictions_before.labels = dataset_test_preds_before

    # Initialize BinaryLabelMetric and ClassifiedMetric
    binary_label_metric = init_binary_label_dataset_metric(L, unprivileged_group, privileged_group)

    classified_metric = init_classification_metric(test, predictions_before, unprivileged_group,
                                                   privileged_group)

    # Compute Fairness Metrics
    # Statistical Parity is the only Fairness Metric that only evaluates fairness in data
    statistical_parity = get_statistical_parity(binary_label_metric)
    equalized_odds = get_equalized_odds(classified_metric)
    equal_opportunity = classified_metric.equal_opportunity_difference()
    false_positive_error_rate_balance = get_false_positive_error_rate_balance(classified_metric)
    predictive_parity = get_predictive_parity(classified_metric)

    return statistical_parity, equalized_odds, equal_opportunity, false_positive_error_rate_balance, predictive_parity


def init_binary_label_dataset_metric(dataset, unprivileged_groups, privileged_groups):
    metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    return metric


def init_classification_metric(ground_truth, predictions, unprivileged_groups, privileged_groups):
    metric = ClassificationMetric(ground_truth, predictions, unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
    return metric


def get_statistical_parity(binary_label_metric):
    statistical_parity = binary_label_metric.mean_difference()
    return statistical_parity


def get_equalized_odds(classified_metric):
    equalized_odds = classified_metric.average_odds_difference()
    return equalized_odds


def get_equal_opportunity(classified_metric):
    equal_opportunity = classified_metric.equal_opportunity_difference()
    return equal_opportunity


def get_false_positive_error_rate_balance(classified_metric):
    false_positive_error_rate_balance = classified_metric.false_positive_rate_difference()
    return false_positive_error_rate_balance


def get_predictive_parity(classified_metric):
    predictive_parity = classified_metric.false_discovery_rate_difference()
    return predictive_parity


def present_comparison(B, B_updated):
    print('***'*35, "\n")
    print("Fairness Metrics before and after updating the model: \n")
    print("Statistical Parity before updating: ", B[0],
          "\n Statistical Parity after updating: ", B_updated[0],
          "\n",
          "\n Equalized Odds before updating : ", B[1], "\n Equalized Odds after updating : ", B_updated[1],
          "\n",
          "\n Equal Opportunity before updating: ", B[2], "\n Equal Opportunity after updating: ", B_updated[2],
          "\n",
          "\n False Positive Error Rate Balance before updating: ", B[3],
          "\n False Positive Error Rate Balance after updating: ", B_updated[3],
          "\n",
          "\n Predictive Parity before updating: ", B[4], "\n Predictive Parity after updating: ", B_updated[4],
          "\n")
    print('***'*35, "\n")


def present_metrics(B):
    print('***'*35, "\n")
    print("Fairness Metrics before updating the model: \n")
    print("Statistical Parity: ", B[0], "\n Equalized Odds: ", B[1], "\n Equal Opportunity: ", B[2],
          "\n False Positive Error Rate Balance: ", B[3], "\n Predictive Parity: ", B[4])
    print('***'*35, "\n")
