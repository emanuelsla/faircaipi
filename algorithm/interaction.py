import numpy

from algorithm.biased_decision import is_relevant, is_in_DN_or_FP


def to_counter_examples(shap_values: list, x_query: numpy.ndarray, x_idx: numpy.ndarray, protected_idx: numpy.ndarray,
                        y_hat: numpy.ndarray, y_U: numpy.ndarray, unfavorable_label: int,
                        min_perc: float) -> (numpy.ndarray, numpy.ndarray):
    """
    Handles label and explanation corrections in interaction mode according to the three cases WWR, RWR and RRR and
    criteria for (un)fair explanations. If the user corrects a label (WWR), no explanation correction is performed.
    If no label correction is performed, an explanation correction is expected to be performed only if the explanation
    is unfair. For an explanation to be unfair the protected attribute must be relevant in the explanation and the
    discriminating groups DN or FP must be present. If an explanation correct is performed the case RWR is present. If
    no explanation correction is performed, case RRR is present.

    :param shap_values: list of SHAP values
    :type shap_values: list
    :param x_query: instance values of query x in unlabeled data pool U
    :type x_query: numpy.ndarray
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
    if ask_user("Would you like to correct the prediction? (y/n) "):
        # WWR: label correction
        c, c_label = correct_prediction(x_query, y_hat)
        present_label_correction(y_hat, y_U, x_idx)
    else:
        if is_relevant(shap_values, x_idx, protected_idx, unfavorable_label, min_perc):
            if is_in_DN_or_FP(x_query, x_idx, shap_values, protected_idx, unfavorable_label):  # bias is present since groups DN and FP are present
                # RWR: explanation correction expected
                if ask_user("Would you like to correct the explanation? (y/n) "):
                    c, c_label = correct_explanation(x_query, protected_idx, y_hat)
                    present_explanation_correction(x_query, y_hat, c, c_label)
                else:
                    # no correction although required
                    if notify_bias(True):
                        # passive_bias_counter = passive_bias_counter + 1
                        c, c_label = x_query, y_hat
                        present_no_correction(x_query, y_hat)
                    else:
                        c, c_label = correct_explanation(x_query, protected_idx, y_hat)
                        present_explanation_correction(x_query, y_hat, c, c_label)
            else:  # bias is not present since groups DP and FN are present
                # RRR: no correction expected
                if ask_user("Would you like to correct the explanation? (y/n) "):
                    if notify_bias(False):
                        # correction leads to an even more biased model
                        # active_bias_counter = active_bias_counter + 1
                        c, c_label = correct_explanation(x_query, protected_idx, y_hat)
                        present_explanation_correction(x_query, y_hat, c, c_label)
                    else:
                        c, c_label = x_query, y_hat
                        present_no_correction(x_query, y_hat)
                else:
                    c, c_label = x_query, y_hat
                    present_no_correction(x_query, y_hat)
        else:  # bias is not present since s is not relevant in the explanation
            # RRR
            print("Since the label is correct and the protected attribute is not relevant in the explanation "
                  "correcting the query does not make sense.")
            if ask_user("Would you still like to correct the explanation? (y/n) "):
                c, c_label = correct_explanation(x_query, protected_idx, y_hat)
                present_explanation_correction(x_query, y_hat, c, c_label)
            else:
                c, c_label = x_query, y_hat
                present_no_correction(x_query, y_hat)
    return c, c_label


def accept_string(inp: str):
    if inp.lower() in ('yes', 'true', 'y', '1'):
        return True
    elif inp.lower() in ('no', 'false', 'n', '0'):
        return False
    else:
        return None


def ask_user(msg=None):
    while True:
        inp = input(msg)
        ret = accept_string(inp)
        if ret is None:
            continue
        else:
            return ret


def get_label_description(label: numpy.ndarray):
    if label[0] == 0:
        risk = 'Good Credit Risk'
    elif label[0] == 1:
        risk = 'Bad Credit Risk'
    else:
        risk = 'Something went wrong, are the labels still at 1 and 2?'
    return risk


def correct_explanation(x_query: numpy.ndarray, protected_idx: int,
                        y_hat: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    counterexample = x_query.copy()
    # since protected attribute is always binary, inverse value of s is sufficient
    counterexample[protected_idx] = int(not (x_query[protected_idx]))
    return counterexample, y_hat


def correct_prediction(x_query: numpy.ndarray, y_hat: numpy.ndarray) -> (numpy.ndarray, numpy.ndarray):
    y = int(not y_hat)
    return x_query, [y]


def present_explanation_correction(x_query: numpy.ndarray, y_hat: numpy.ndarray, counterexample_x: numpy.ndarray,
                                   counterexample_y: numpy.ndarray):
    print('***' * 35, "\n")
    print("Fair explanation correction from query \n", x_query, "\n with label", y_hat, "--",
          get_label_description(y_hat), "to counterexamples \n", counterexample_x, "\n with label",
          counterexample_y, "--", get_label_description(counterexample_y), ".\n")
    print('***' * 35, "\n")


def present_label_correction(y_hat: numpy.ndarray, y_U: numpy.ndarray, x_idx: numpy.ndarray):
    y = int(not y_hat)
    print('***' * 35, "\n")
    print("Label correction from prediction", y_hat, "--", get_label_description(y_hat), "to label", [y], "--",
          get_label_description([y]), "with ground truth label", y_U[x_idx], "--", get_label_description(y_U[x_idx]), ".\n")
    print('***' * 35, "\n")


def present_no_correction(x_query: numpy.ndarray, y_hat: numpy.ndarray):
    print('***' * 35, "\n")
    print("No correction. Model is updated on original query \n", x_query, "\n with label", y_hat, "--",
          get_label_description(y_hat), ".\n")
    print('***' * 35, "\n")


def notify_bias(passive: bool):
    if passive:
        return ask_user("Your decision not to correct the query might be biased. Are you sure you want to continue "
                        "without correction? (y/n) ")
    else:
        return ask_user("Your decision to correct the query might be biased. Are you sure you want to continue and "
                        "correct the query? (y/n) ")


def is_f_fair_enough():
    print("If you consider f fair enough, you can stop the correction process.")
    return ask_user("Would like to stop the correction process since f is fair enough? (y/n) ")


def check_accuracy(a: float, Z: float):
    print("The model's accuracy is ", a, ". The accuracy threshold is ", Z, ".")
    return ask_user("Would you like to stop with respect to the accuracy ? (y/n) ")
