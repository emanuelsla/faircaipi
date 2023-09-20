import aif360
from modAL.models import ActiveLearner
import csv

from sklearn.base import BaseEstimator

from algorithm import active_learner as al
from ai_fairness_360.dataset import split_data_into_X_and_y
from algorithm.correlations import compute_correlations, present_correlations
from explainers.shap_explainer import generate_explainer, display_explanation
from simulation_experiment import simulation
from simulation_experiment.simulation import is_pred_correct, \
    get_bias_mask, compute_fair_explanation_metrics, \
    get_DN_or_FP_mask, count_explanations
from algorithm.biased_decision import is_biased
from algorithm import interaction
from algorithm.interaction import is_f_fair_enough, check_accuracy
from ai_fairness_360.fairness_metrics import present_metrics, present_comparison, compute_fairness_metrics
from algorithm.update_pools import update_U, update_L
from algorithm.active_learner import present_query
from sklearn.metrics import classification_report


def fair_CAIPI(L: aif360.datasets.Dataset, U: aif360.datasets.Dataset, test: aif360.datasets.Dataset, Z: float, T: int,
               s: str, estimator: BaseEstimator, min_perc: float, data_file: str, interactive: bool) -> ActiveLearner:
    """
    Fair Interacting with Explanations, starts the fairCAIPI process for correcting labels
    and explanations in terms of fairness in either interaction or simulation mode. In interaction mode the user is
    asked to correct labels and explanations in terms of fairness in order to teach fairness to the model. In simulation
    mode corrections are made automatically.

    :param L: labeled data pool in AIF360 StandardDataset format on which the model is trained on
    :type L: aif360.datasets.Dataset
    :param U: unlabeled data pool in AIF360 StandardDataset format the query is selected from
    :type U: aif360.datasets.Dataset
    :param test: separate test set in AIF360 StandardDataset format that allows to evaluate the model
    :type test: aif360.datasets.Dataset
    :param Z: accuracy threshold as orientation for user when interacting with the model
    :type Z: float
    :param T: iteration budget for simulation
    :type T: int
    :param s: selected binary protected attribute, e.g. 'sex', 'age', 'race' or other
    :type s: str
    :param estimator: sklearn base estimator that functions as the underlying model of the ActiveLearner
    :type estimator: BaseEstimator
    :param min_perc: percentage value, that indicates which labels do not contribute at least minimum percentage to the total
    :type min_perc: float
    :param data_file: data file to print simulation results in
    :type data_file: str
    :param interactive: determines whether interaction (True) or simulation mode (False) is started
    :type interactive: bool

    :return: (Fairer) ActiveLearner model
    """

    to_counter_examples = interaction.to_counter_examples if interactive else simulation.to_counter_examples

    if not interactive:
        # Initialization of writer and file for writing data to CSV file
        file = open(data_file, 'w')
        writer = csv.writer(file)

    # Prepare data
    X_L, y_L = split_data_into_X_and_y(L)
    X_U, y_U = split_data_into_X_and_y(U)
    X_test, y_test = split_data_into_X_and_y(test)

    feature_names = L.convert_to_dataframe()[0].columns[:-1]
    protected_idx = L.feature_names.index(s)
    unfavorable_label = not U.favorable_label

    # f = Fit(L)
    f: ActiveLearner = al.fit(estimator, X_L, y_L)

    # Accuracy of initialized f
    a = f.score(X_test, y_test)

    pred = f.predict(X_test)

    classification = classification_report(y_test, pred)

    print("Classification report before interaction: \n", classification)

    if not interactive:
        i = 0
        biased_explanation_counter = 0
        label_corrections_counter = 0

        explainer_L, shap_values_L = generate_explainer(f, X_L)
        explainer_test, shap_values_test = generate_explainer(f, X_test)

        mean_diff_L = 0
        mean_diff_test = 0

        mean_L = 0
        mean_test = 0

    else:
        print("The model is now fit with the initial training data pool L. It achieves an initial accuracy of ", a, ".")

    while True:

        # R = ComputeCorrelations(s,L)
        R = compute_correlations(s, L)

        if interactive:
            # B = ComputeFairnessMetric(f,L)
            B = compute_fairness_metrics(f, test, L, s)
            # Present B to the user
            present_metrics(B)
            # Present R and s to the user
            present_correlations(s, R)

        # x = SelectQuery(f,U)
        if len(X_U) > 0:
            x_idx, x_query = al.select_query(f, X_U)
        else:
            print("Since U is empty, it is impossible to correct another query. The interaction process is stopped.")
            break
        # y_hat = f(x) → prediction
        y_hat = f.predict(X_U[x_idx])
        # true label y
        y = y_U[x_idx]

        # explain query x
        explainer, shap_values = generate_explainer(f, X_U)

        # shap_value_protected = shap_values[1][x_idx, :][0][protected_idx]
        if interactive:
            # Present z_hat, x and y_hat to user (explanation for category: 'Bad Credit Risk')
            present_query(U, x_idx)
            display_explanation(explainer, X_U, y_U, U, y_hat, shap_values, x_idx, feature_names)

        # Obtain y and explanation correction C with respect to relevance of s and group membership
        # if decision is biased then notify user and present x, y_hat and z_hat again
        # if any obtain, y and C

        # {(x,y)} = ToCounterExamples(C)
        counterexample_x, counterexample_y = to_counter_examples(shap_values, x_query, x_idx, protected_idx, y_hat, y_U,
                                                                 unfavorable_label, min_perc)

        if not interactive:
            if is_pred_correct(y_hat, y_U, x_idx) and is_biased(shap_values, x_query, x_idx, protected_idx,
                                                                unfavorable_label, min_perc):
                biased_explanation_counter = biased_explanation_counter + 1
            if not is_pred_correct(y_hat, y_U, x_idx):
                label_corrections_counter = label_corrections_counter + 1

        # Update L
        L, X_L, y_L = update_L(X_L, y_L, L, counterexample_x, counterexample_y)

        # Update U
        U, X_U, y_U = update_U(X_U, y_U, U, x_idx)

        # Update f
        f.teach([counterexample_x], counterexample_y)

        if not interactive:
            # Compute mean Shap values of the protected attribute for L and test set and mean difference of Shap values
            # of the protected attribute from previous and current iteration

            pre_shap_values_L = shap_values_L
            pre_shap_values_test = shap_values_test

            explainer_L, shap_values_L = generate_explainer(f, X_L)
            explainer_test, shap_values_test = generate_explainer(f, X_test)

            bias_mask_L = get_bias_mask(shap_values_L, X_L, protected_idx, unfavorable_label, min_perc)
            bias_mask_test = get_bias_mask(shap_values_test, X_test, protected_idx, unfavorable_label, min_perc)
            unfair_explanation_counter_L = count_explanations(bias_mask_L)
            unfair_explanation_counter_test = count_explanations(bias_mask_test)

            DN_FP_mask_L = get_DN_or_FP_mask(shap_values_L, X_L, protected_idx, unfavorable_label)
            DN_FP_mask_test = get_DN_or_FP_mask(shap_values_test, X_test, protected_idx, unfavorable_label)
            DN_FP_counter_L = count_explanations(DN_FP_mask_L)
            DN_FP_counter_test = count_explanations(DN_FP_mask_test)

            mean_diff_L, mean_diff_test, mean_L, mean_test = compute_fair_explanation_metrics(shap_values_L,
                                                                                              pre_shap_values_L,
                                                                                              shap_values_test,
                                                                                              pre_shap_values_test,
                                                                                              X_L, X_test,
                                                                                              protected_idx,
                                                                                              unfavorable_label)

        # B_updated = ComputeFairnessMetric(f,L)
        B_updated = compute_fairness_metrics(f, test, L, s)

        # a = ComputeAccuracy(f,testset)
        a = f.score(X_test, y_test)

        if interactive:
            # Present comparison of B and B_updated
            present_comparison(B, B_updated)
        else:
            # Write iterations, accuracy threshold, accuracy and fairness metrics to CSV file
            writer.writerow((i, Z, a, biased_explanation_counter, label_corrections_counter,
                             unfair_explanation_counter_L, unfair_explanation_counter_test, DN_FP_counter_L,
                             DN_FP_counter_test, mean_diff_L, mean_diff_test, mean_L, mean_test,) + B_updated)

        # stop interaction process or continue
        if interactive:
            if is_f_fair_enough() or check_accuracy(a, Z):  # since there is no do while loop built-in in Python
                break
        else:
            i = i + 1
            if i >= T:
                break

    pred_after = f.predict(X_test)

    present_metrics(B_updated)

    classification_after = classification_report(y_test, pred_after)

    print("Classification report after interaction: \n", classification_after)

    if not interactive:
        file.close()

    return f
