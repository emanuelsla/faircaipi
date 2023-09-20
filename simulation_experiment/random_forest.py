import aif360 as aif360
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from ai_fairness_360.dataset import split_data_into_X_and_y
from ai_fairness_360.fairness_metrics import init_binary_label_dataset_metric, init_classification_metric, \
    get_statistical_parity, get_equalized_odds, get_false_positive_error_rate_balance, get_predictive_parity
from ai_fairness_360.reweighing import reweigh_dataset


def evaluate_default_rf(train: aif360.datasets.Dataset, test: aif360.datasets.Dataset, unprivileged_group: dict,
                        privileged_group: dict):
    """
    Shows evaluation results for default Random Forest

    :param train: train set to initialize rf
    :type train: aif360.datasets.Dataset
    :param test: test set to evaluate rf
    :type test: aif360.datasets.Dataset
    :param unprivileged_group: Mapping of protected group to integers
    :type unprivileged_group: dict
    :param privileged_group: Mapping of privileged group to integers
    :type privileged_group: dict
    :return:
    """
    X_train, y_train = split_data_into_X_and_y(train)
    X_test, y_test = split_data_into_X_and_y(test)

    # Initialize default RF with original train data
    default_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    default_rf.fit(X_train, y_train)

    y_predict_default = default_rf.predict(X_test)

    # Accuracy of default RF
    classification_rf_default = classification_report(y_test, y_predict_default)
    print("Classification report of default RF: \n", classification_rf_default)

    # Get predictions from default RF
    dataset_test_preds_before = default_rf.predict(test.features)
    dataset_test_preds_prob_before = default_rf.predict_proba(test.features)[:, 1]
    predictions_before_default = test.copy()
    predictions_before_default.scores = dataset_test_preds_prob_before
    predictions_before_default.labels = dataset_test_preds_before

    # Initialize BinaryLabelMetric and ClassifiedMetric for original data and default RF
    # For BinaryLabelMetric train set
    binary_label_metric_before = init_binary_label_dataset_metric(train, unprivileged_groups=[unprivileged_group],
                                                                  privileged_groups=[privileged_group])
    # For ClassificationMetric test set
    classified_metric_before = init_classification_metric(test, predictions_before_default,
                                                          unprivileged_groups=[unprivileged_group],
                                                          privileged_groups=[privileged_group])

    statistical_parity = get_statistical_parity(binary_label_metric_before)
    print("Statistical Parity of original Dataset:", '{:f}'.format(statistical_parity))
    equalized_odds = get_equalized_odds(classified_metric_before)
    print("Equalized Odds for RandomForest trained on original Dataset:", '{:f}'.format(equalized_odds))
    equal_opportunity = classified_metric_before.equal_opportunity_difference()
    print("Equal Opportunity for RandomForest trained on original Dataset:", '{:f}'.format(equal_opportunity))
    false_positive_error_rate_balance = get_false_positive_error_rate_balance(classified_metric_before)
    print("False Positive Error Rate Balance for RandomForest trained on original Dataset:",
          '{:f}'.format(false_positive_error_rate_balance))
    predictive_parity = get_predictive_parity(classified_metric_before)
    print("Predictive Parity for RandomForest trained on original Dataset::", '{:f}'.format(predictive_parity))


def evaluate_reweighed_rf(train: aif360.datasets.Dataset, test: aif360.datasets.Dataset, unprivileged_group: dict,
                          privileged_group: dict):
    """
    Shows evaluation results for reweighed Random Forest

    :param train: train set that will be reweighed and used to initialize rf
    :type train: aif360.datasets.Dataset
    :param test: test set to evaluate rf
    :type test: aif360.datasets.Dataset
    :param unprivileged_group: Mapping of protected group to integers
    :type unprivileged_group: dict
    :param privileged_group: Mapping of privileged group to integers
    :type privileged_group: dict
    :return:
    """
    X_test, y_test = split_data_into_X_and_y(test)

    train_reweighed = reweigh_dataset(train, unprivileged_group, privileged_group)

    X_train_rw, y_train_rw = split_data_into_X_and_y(train_reweighed)
    reweighed_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    reweighed_rf.fit(X_train_rw, y_train_rw, sample_weight=train_reweighed.instance_weights)

    y_predict = reweighed_rf.predict(X_test)

    # Accuracy of RF trained on reweighed data
    classification_rf = classification_report(y_test, y_predict)
    print("Classification report of RF trained on reweighed data: \n", classification_rf)

    # Get predictions from RF trained on reweighed data
    dataset_test_preds_before = reweighed_rf.predict(test.features)
    dataset_test_preds_prob_before = reweighed_rf.predict_proba(test.features)[:, 1]
    predictions_before = test.copy()
    predictions_before.scores = dataset_test_preds_prob_before
    predictions_before.labels = dataset_test_preds_before

    # Initialize BinaryLabelMetric and ClassifiedMetric for reweighed data and RF
    binary_label_metric = init_binary_label_dataset_metric(train_reweighed, unprivileged_groups=[unprivileged_group],
                                                           privileged_groups=[privileged_group])

    classified_metric = init_classification_metric(test, predictions_before, unprivileged_groups=[unprivileged_group],
                                                   privileged_groups=[privileged_group])

    statistical_parity = get_statistical_parity(binary_label_metric)
    print("Statistical Parity for reweighed Dataset train:", '{:f}'.format(statistical_parity))
    equalized_odds = get_equalized_odds(classified_metric)
    print("Equalized Odds for reweighed RandomForest:", '{:f}'.format(equalized_odds))
    equal_opportunity = classified_metric.equal_opportunity_difference()
    print("Equal Opportunity for reweighed RandomForest:", '{:f}'.format(equal_opportunity))
    false_positive_error_rate_balance = get_false_positive_error_rate_balance(classified_metric)
    print("False Positive Error Rate Balance for reweighed RandomForest:",
          '{:f}'.format(false_positive_error_rate_balance))
    predictive_parity = get_predictive_parity(classified_metric)
    print("Predictive Parity for reweighed RandomForest:", '{:f}'.format(predictive_parity))
