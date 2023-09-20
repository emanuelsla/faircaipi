import aif360.datasets
import numpy
import numpy as np
import pandas as pd
from aif360.datasets import StandardDataset


def update_L(X_L: numpy.ndarray, y_L: numpy.ndarray, L: aif360.datasets.Dataset, counterexample_x: numpy.ndarray,
             counterexample_y: list):
    """
    Updates L by adding counterexamples

    :param X_L: features of L without labels
    :type X_L: numpy.ndarray
    :param y_L: labels of L
    :type y_L: numpy.ndarray
    :param L: complete L in StandardDataset fromat
    :type L: aif360.datasets.Dataset
    :param counterexample_x:
    :type counterexample_x: numpy.ndarray
    :param counterexample_y:
    :type counterexample_y: numpy.ndarray
    :return: updated StandardDataset L, updated numpy.ndarray X_L, updated numpy.ndarray y_L
    """
    X_L_updated, y_L_updated = update_train_data(X_L, y_L, counterexample_x, counterexample_y)
    # add instance to L (type StandardDataset): Only way is to convert StandardDataset type into DataFrame and back
    label_name = L.label_names
    protected_attribute = L.protected_attribute_names
    privileged_group = L.privileged_protected_attributes
    favorable_class = L.favorable_label

    counterexample_df = np.append(counterexample_x, counterexample_y)

    df = pd.DataFrame(counterexample_df.reshape(-1, len(counterexample_x) + 1),
                      columns=['credit_amount', 'age', 'number_of_credits', 'skill', 'sex',
                               'status=200+', 'status=<200', 'status=None', 'credit_history=Critical',
                               'credit_history=Delay', 'credit_history=None/Paid', 'purpose=Business',
                               'purpose=Car', 'purpose=Education', 'purpose=Other', 'savings=500+',
                               'savings=<500', 'savings=Unknown/None', 'employment=1-4 years',
                               'employment=4+ years', 'employment=Unemployed', 'property=Other',
                               'property=Real Estate', 'property=Unknown/No Property',
                               'housing=For free', 'housing=Own', 'housing=Rent', 'credit'])
    df_L = L.convert_to_dataframe()[0]
    df_L = pd.concat([df_L, df])

    L_updated = StandardDataset(df_L, label_name=label_name[0], favorable_classes=[favorable_class],
                                protected_attribute_names=protected_attribute,
                                privileged_classes=privileged_group,
                                metadata={'label_map': {0.0: 'Good Credit', 1.0: 'Bad Credit'},  # TODO do not hardcode
                                          'protected_attribute_maps': [{1.0: 'Male', 0.0: 'Female'}]})

    return L_updated, X_L_updated, y_L_updated


def update_train_data(X_L: numpy.ndarray, y_L: numpy.ndarray, counterexample_x: numpy.ndarray, counterexample_y: list):
    X_L_updated = np.concatenate((X_L, [counterexample_x]), axis=0)
    y_L_updated = np.concatenate((y_L, counterexample_y), axis=0)
    print('***'*35, "\n")
    print("L is now updated and changed in size from shape", X_L.shape, "to", X_L_updated.shape, ".\n")
    print('***'*35, "\n")
    return X_L_updated, y_L_updated


def update_U(X_U: numpy.ndarray, y_U: numpy.ndarray, U: aif360.datasets.Dataset, x_idx: numpy.ndarray):
    """
    Updates U by removing query x

    :param X_U: features of U without labels
    :type X_U: numpy.ndarray
    :param y_U: labels of U
    :type y_U: numpy.ndarray
    :param U: complete U in StandardDataset fromat
    :type U: aif360.datasets.Dataset
    :return: updated StandardDataset U, updated numpy.ndarray X_U, updated numpy.ndarray y_U

    """

    X_test_updated, y_test_updated = update_test_data(X_U, y_U, x_idx)
    # remove instance from U (type StandardDataset): AIF dataset only allows to use the subset function
    U_updated = U.subset(list(filter(lambda x: x != int(x_idx[0]), range(0, U.features.shape[0]))))
    return U_updated, X_test_updated, y_test_updated


def update_test_data(X_U: numpy.ndarray, y_U: numpy.ndarray, x_idx: numpy.ndarray):
    X_test_updated = np.delete(X_U, (x_idx[0]), axis=0)
    y_test_updated = np.delete(y_U, [x_idx[0]], axis=0)
    print('***'*35, "\n")
    print("U is now updated and changed in size from shape", X_U.shape, "to", X_test_updated.shape, ".\n")
    print('***'*35, "\n")
    return X_test_updated, y_test_updated
