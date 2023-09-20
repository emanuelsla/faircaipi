from ai_fairness_360.dataset import dataset_as_dataframe
import aif360


def compute_correlations(s: str, L: aif360.datasets.Dataset):
    """
    Computes correlations of protected attribute s with all other unprotected attributes in data set L

    :param s: selected binary protected attribute, e.g. 'sex', 'age', 'race' or other
    :type s: str
    :param L: labeled data pool in AIF360 StandardDataset format on which the model is trained on
    :type L: aif360.datasets.Dataset
    :return: pandas.core.series.Series of correlation values
    """
    L_as_df = dataset_as_dataframe(L)
    return L_as_df.corrwith(L_as_df[s])


def present_correlations(s, R):
    print('***' * 35, "\n")
    print(print("Attributes correlating with protected attribute", s, ": \n"))
    print(R, "\n")
    print('***' * 35, "\n")
