import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv(file_name: str):
    df = pd.read_csv(file_name)
    df.columns = ["iteration", "accuracy_threshold", "accuracy", "explanation_corrections",
                  "label_corrections", "unfair_explanations_L", "unfair_explanations_test", "DN_FP_counter_L",
                  "DN_FP_counter_test", "mean_diff_shap_values_L", "mean_diff_shap_values_test", "mean_shap_values_L",
                  "mean_shap_values_test", "statistical_parity", "equalized_odds", "equal_opportunity",
                  "false_positive_error_rate_balance", "predictive_parity"]
    print(df)
    return df


def plot_fairness_metrics(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, 0.5)  # (-1,1)

    plt.axhline(y=0, xmin=0, xmax=100, color='black')
    ax.plot(df['iteration'], df['statistical_parity'])
    ax.plot(df['iteration'], df['equalized_odds'])
    ax.plot(df['iteration'], df['equal_opportunity'])
    ax.plot(df['iteration'], df['false_positive_error_rate_balance'])
    ax.plot(df['iteration'], df['predictive_parity'])

    ax.set_title("Fairness Metrics in Comparison")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Fairness Metrics")
    fig.legend(
        ['Optimum', 'Statistical Parity', 'Equalized Odds', 'Equal Opportunity', 'False Positive Error Rate Balance',
         'Predictive Parity'])
    plt.savefig("simulation_plots/metrics.svg")
    plt.show()


def plot_accuracy(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1)

    ax.plot(df['iteration'], df['accuracy'], linestyle='dashed')
    ax.plot(df['iteration'], df['accuracy_threshold'])

    # ax.set_title("Accuracy")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Accuracy")
    fig.legend(['Accuracy for test set', 'Accuracy Threshold'])
    plt.savefig("simulation_plots/accuracy.svg")
    plt.show()


def plot_corrections(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    # ax.set_ylim()

    ax.plot(df['iteration'], df['explanation_corrections'])
    ax.plot(df['iteration'], df['label_corrections'])

    # ax.set_title("Accuracy")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Counts")
    fig.legend(['Number of Explanation Corrections', 'Number of Label Corrections'])
    plt.savefig("simulation_plots/count.svg")
    plt.show()


def plot_unfair_explanations(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    ax.set_ylim(0, 600)

    ax.plot(df['iteration'], df['unfair_explanations_L'])
    ax.plot(df['iteration'], df['unfair_explanations_test'])
    ax.plot(df['iteration'], df['DN_FP_counter_L'])
    ax.plot(df['iteration'], df['DN_FP_counter_test'])

    # ax.set_title("Accuracy")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Counts")
    fig.legend(['Number of Unfair Explanations in L', 'Number of Unfair Explanations in test set',
                'Number of DN or FP group in L', 'Number of DN or FP group in test set'])
    plt.savefig("simulation_plots/unfair_explanations.svg")
    plt.show()


def plot_mean_diff_shap_values(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    ax.set_ylim(-0.005, 0.005)

    ax.plot(df['iteration'], df['mean_diff_shap_values_L'])
    ax.plot(df['iteration'], df['mean_diff_shap_values_test'])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean difference in shap values of protected attribute 'sex'")
    fig.legend(['Mean difference in Shap values for L', 'Mean difference in Shap values for test set'])
    plt.savefig("simulation_plots/mean_diff.svg")
    plt.show()


def plot_mean_shap_values(df: pandas.DataFrame, T: int):
    fig, ax = plt.subplots()
    ax.set_xlim(0, T)
    ax.set_ylim(0, 0.05)

    ax.plot(df['iteration'], df['mean_shap_values_L'])
    ax.plot(df['iteration'], df['mean_shap_values_test'])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Mean of Shap values of protected attribute 'sex'")
    fig.legend(['Mean of Shap values for L', 'Mean of Shap values for test set'])
    plt.savefig("simulation_plots/mean.svg")
    plt.show()


def compute_mean(df, column):
    return np.mean(df[column])


def closest_to_zero(df, column):
    min_value = min(abs(df[column]))
    minimum = 0, 0
    for idx, value in df[column].iteritems():
        if abs(value) == min_value:
            minimum = idx, value
    return minimum


def max_value(df, column):
    max_value = max(df[column])
    maximum = 0, 0
    for idx, value in df[column].iteritems():
        if value == max_value:
            maximum = idx, value
    return maximum