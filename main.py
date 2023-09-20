import datetime
import argparse
import pandas as pd
from aif360.datasets import StandardDataset
import ai_fairness_360
from ai_fairness_360.german_dataset import init_german_dataset
from algorithm.fair_caipi import fair_CAIPI
from sklearn.ensemble import RandomForestClassifier
from simulation_experiment.plots import read_csv, plot_fairness_metrics, plot_accuracy, \
    plot_mean_shap_values, plot_mean_diff_shap_values, compute_mean, closest_to_zero, max_value, plot_corrections, \
    plot_unfair_explanations
from simulation_experiment.random_forest import evaluate_default_rf, evaluate_reweighed_rf

if __name__ == '__main__':

    def parse_args():
        mvar = '?'
        parser = argparse.ArgumentParser(description='Fair Interacting with Explanations')
        parser.add_argument('--seed', default=42, type=int,
                            help='seed to use for randomization', metavar=mvar)
        parser.add_argument('--Z', default=0.7, type=float,
                            help='threshold under which the accuracy should not fall during the interaction process '
                                 'for orientation')
        parser.add_argument('--min_perc', default=0.005, type=float,
                            help='minimum percentage to determine contribution threshold of protected attribute '
                                 'Shapley value')
        parser.add_argument('--T', default=100, type=int,
                            help='number of iterations')

        subparsers = parser.add_subparsers(help='sub commands', dest='subcommand')
        # interactive
        parser_interactive = subparsers.add_parser('interactive', help='run interactive mode')
        # simulation
        parser_simulation = subparsers.add_parser('simulation',
                                                  help='run simulation mode')
        parser_simulation.add_argument('--print_plot', default=False, type=bool, metavar=mvar,
                                       help='print plot of data generated during simulation')
        # plot
        parser_plot = subparsers.add_parser('plot',
                                            help='print plots from existing csv-file')
        parser_plot.add_argument('--plot_file', required=True, type=str, metavar=mvar,
                                 help='path to csv-file')
        # default_rf
        parser_default_rf = subparsers.add_parser('default_rf',
                                                  help='show evaluation results for default Random Forest')

        # reweighed_rf
        parser_reweighed_rf = subparsers.add_parser('reweighed_rf',
                                                    help='show evaluation results for reweighed Random Forest')

        return parser.parse_args()


    cfg = parse_args()

    if cfg.subcommand in ["simulation", "interactive", "default_rf", "reweighed_rf"]:
        # Initialize German Credit Approval Dataset with AIF360

        protected_attribute = 'sex'
        favorable_class = [0]  # 'Good Credit'
        privileged_class = [[1.0]]  # 'Male'
        label_map = {0.0: 'Good Credit', 1.0: 'Bad Credit'}
        protected_attribute_maps = {1.0: 'Male', 0.0: 'Female'}

        german_dataset = init_german_dataset(protected_attribute, privileged_class, favorable_class, label_map,
                                             protected_attribute_maps)

        # split into L, U and test set
        train, test = german_dataset.split([0.7], shuffle=True, seed=cfg.seed)
        L, U = train.split([0.786], shuffle=True, seed=cfg.seed)

        # set accuracy threshold
        Z = cfg.Z

        # initialize estimator with Random Forest
        estimator = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=cfg.seed)

        # For default Random Forest and reweighed Random Forest
        privileged_protected_value = train.privileged_protected_attributes[0][0]
        unprivileged_groups = {protected_attribute: int(not privileged_protected_value)}
        privileged_groups = {protected_attribute: privileged_protected_value}

    # set iteration budget
    T = cfg.T
    # name of csv file
    ct = datetime.datetime.now()
    data_file = f"simulation_files/simulation_data_{ct}.csv"
    min_perc = cfg.min_perc
    if cfg.subcommand == 'interactive':
        model = fair_CAIPI(L, U, test, Z, T, protected_attribute, estimator, min_perc, data_file=data_file,
                           interactive=True)
    elif cfg.subcommand == 'simulation':
        model = fair_CAIPI(L, U, test, Z, T, protected_attribute, estimator, min_perc, data_file=data_file,
                           interactive=False)
    elif cfg.subcommand == 'plot':
        data_file = cfg.plot_file
    elif cfg.subcommand == 'default_rf':
        # randomly split off 100 unseen data instances from U
        U_100, U_50 = U.split([0.67], shuffle=True, seed=cfg.seed)

        df_U = U_100.convert_to_dataframe()[0]
        df_L = L.convert_to_dataframe()[0]
        df_L = pd.concat([df_L, df_U])

        L_U_100 = StandardDataset(df_L, label_name=L.label_names[0], favorable_classes=favorable_class,
                                  protected_attribute_names=[protected_attribute],
                                  privileged_classes=privileged_class,
                                  metadata={'label_map': {0.0: 'Good Credit', 1.0: 'Bad Credit'},
                                            'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]})
        evaluate_default_rf(L_U_100, test, unprivileged_groups, privileged_groups)
    elif cfg.subcommand == 'reweighed_rf':
        train_reweighed = ai_fairness_360.reweighing.reweigh_dataset(train, unprivileged_groups, privileged_groups)

        L_reweighed, U_reweighed = train_reweighed.split([0.786], shuffle=True, seed=cfg.seed)
        # randomly split off 100 unseen data instances from U
        U_100_reweighed, U_50_reweighed = U_reweighed.split([0.67], shuffle=True, seed=cfg.seed)

        df_U = U_100_reweighed.convert_to_dataframe()[0]
        df_L = L_reweighed.convert_to_dataframe()[0]
        df_L = pd.concat([df_L, df_U])

        L_U_100_reweighed = StandardDataset(df_L, label_name=L.label_names[0], favorable_classes=favorable_class,
                                            protected_attribute_names=[protected_attribute],
                                            privileged_classes=privileged_class,
                                            metadata={'label_map': {0.0: 'Good Credit', 1.0: 'Bad Credit'},
                                                      'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]})

        evaluate_reweighed_rf(L_U_100_reweighed, test, unprivileged_groups, privileged_groups)

    if (cfg.subcommand == 'simulation' and cfg.print_plot) or cfg.subcommand == 'plot':
        df = read_csv(data_file)
        plot_fairness_metrics(df, T)
        plot_accuracy(df, T)
        plot_corrections(df, T)
        plot_unfair_explanations(df, T)
        plot_mean_diff_shap_values(df, T)
        plot_mean_shap_values(df, T)

        print("Mean of Statistical Parity:", compute_mean(df, 'statistical_parity'))
        print("Optimum of Statistical Parity:", closest_to_zero(df, 'statistical_parity'))
        print(compute_mean(df, 'accuracy'))
        print("Optimum of accuracy:", max_value(df, 'accuracy'))
        print("Optimum of Equal Opportunity:", closest_to_zero(df, 'equal_opportunity'))
        print("Optimum of Equalized Odds:", closest_to_zero(df, 'equalized_odds'))
        print("Optimum of FPERB:", closest_to_zero(df, 'false_positive_error_rate_balance'))
        print("Optimum of Predictive Parity:", closest_to_zero(df, 'predictive_parity'))

# df = read_csv("simulation_data_2022-09-15 14:54:45_min_perc=0_T=100.csv") # min_perc = 0.00, T=100
# df = read_csv("simulation_data_2022-09-15 15:48:49_min_perc=0.005_T=100.csv") # ! evaluation min_perc = 0.005, T=100
# df = read_csv("simulation_data_2022-09-15 13:24:06_min_perc=0.01_T=100.csv") # min_perc = 0.01, T=100
# df = read_csv("simulation_data_2022-09-15 16:43:01_min_perc=0.015_T=100.csv")  # min_perc = 0.015, T=100
# df = read_csv("simulation_data_2022-09-13 21:52:45_min_perc=0.02_T=100.csv") # min_perc = 0.02, T=100
# df = read_csv("simulation_data_2022-09-14 18:02:05_min_perc=0.05_T=100.csv") # min_perc = 0.05, T=100