import numpy
import pandas
import numpy as np
from aif360.datasets import GermanDataset
from sklearn.preprocessing import MinMaxScaler


def init_german_dataset(protected_attribute: str, privileged_class: numpy.ndarray, favorable_class: numpy.ndarray,
                        label_map: dict, protected_attribute_maps: dict):
    """
    Initializes German Credit Approval Dataset with costumized pre-processing

    :param protected_attribute: selected binary protected attribute, e.g. 'sex', 'age', 'race' or other
    :type protected_attribute: str
    :param privileged_class: privileged class of protected attribute
    :param privileged_class: numpy.ndarray
    :param favorable_class: favorable label
    :type favorable_class: numpy.ndarray
    :param label_map: Mapping of favorable and unfavorable labels to integers
    :type label_map: dict
    :param protected_attribute_maps: Mapping of protected and unprotected attribute to integers
    :type protected_attribute_maps: dict
    :return: aif360.datasets.Dataset (German)
    """
    german_dataset = GermanDataset(
        label_name='credit',
        favorable_classes=favorable_class,
        protected_attribute_names=[protected_attribute],
        privileged_classes=privileged_class,  # which corresponds to 'Male',
        metadata={'label_map': label_map, 'protected_attribute_maps': [protected_attribute_maps]},
        features_to_keep=['credit_amount', 'age', 'skill', 'number_of_credits', 'sex', 'status', 'housing',
                          'credit_history', 'purpose', 'property', 'savings', 'employment', 'skill_level'],
        features_to_drop=['personal_status', 'other_debtors', 'skill_level', 'installment_plans', 'telephone',
                          'foreign_worker'],
        custom_preprocessing=custom_preprocessing
    )
    return german_dataset


def custom_preprocessing(df: pandas.DataFrame):
    """
    Pre-procossess German Credit Dataset
    This code is originally from:
    https://github.com/Trusted-AI/AIF360/blob/746e763191ef46ba3ab5c601b96ce3f6dcb772fd/aif360/algorithms/preprocessing/optim_preproc_helpers/data_preproc_functions.py
    but extended to more features and MinMaxScaling

    :param df: German Dataset in pandas DataFrame type
    :type df: pandas.DataFrame
    """

    def group_credit_hist(x):
        if x in ['A30', 'A31', 'A32']:
            return 'None/Paid'
        elif x == 'A33':
            return 'Delay'
        elif x == 'A34':
            return 'Critical'
        else:
            return 'NA'

    def group_purpose(x):
        if x in ['A40', 'A41']:
            return 'Car'
        elif x in ['A46', 'A48']:
            return 'Education'
        elif x == 'A49':
            return 'Business'
        elif x in ['A42', 'A43', 'A44', 'A45', 'A47', 'A410']:
            return 'Other'
        else:
            return 'NA'

    def group_property(x):
        if x == 'A121':
            return 'Real Estate'
        elif x in ['A122', 'A123']:
            return 'Other'
        elif x == 'A124':
            return 'Unknown/No Property'
        else:
            return 'NA'

    def group_housing(x):
        if x == 'A151':
            return 'Rent'
        elif x == 'A152':
            return 'Own'
        elif x == 'A153':
            return 'For free'
        else:
            return 'NA'

    def group_employ(x):
        if x == 'A71':
            return 'Unemployed'
        elif x in ['A72', 'A73']:
            return '1-4 years'
        elif x in ['A74', 'A75']:
            return '4+ years'
        else:
            return 'NA'

    def group_savings(x):
        if x in ['A61', 'A62']:
            return '<500'
        elif x in ['A63', 'A64']:
            return '500+'
        elif x == 'A65':
            return 'Unknown/None'
        else:
            return 'NA'

    def group_status(x):
        if x in ['A11', 'A12']:
            return '<200'
        elif x in ['A13']:
            return '200+'
        elif x == 'A14':
            return 'None'
        else:
            return 'NA'

    def scale_min_max(feature):
        scaler = MinMaxScaler(copy=False)
        scaled_feature = scaler.fit_transform(df[feature].values[:, np.newaxis])
        return scaled_feature

    target_map = {1.0: 0.0, 2.0: 1.0}
    df['credit'] = df['credit'].replace(target_map)

    skill_level_map = {'A171': 0.0, 'A172': 0.0, 'A173': 1.0, 'A174': 1.0}
    df['skill'] = df['skill_level'].replace(skill_level_map)

    status_map = {'A91': 1.0, 'A93': 1.0, 'A94': 1.0,
                  'A92': 0.0, 'A95': 0.0}
    df['sex'] = df['personal_status'].replace(status_map)

    # group credit history, purpose, savings, property, housing and employment
    df['credit_history'] = df['credit_history'].apply(lambda x: group_credit_hist(x))
    df['purpose'] = df['purpose'].apply(lambda x: group_purpose(x))
    df['savings'] = df['savings'].apply(lambda x: group_savings(x))
    df['property'] = df['property'].apply(lambda x: group_property(x))
    df['housing'] = df['housing'].apply(lambda x: group_housing(x))
    df['employment'] = df['employment'].apply(lambda x: group_employ(x))
    df['status'] = df['status'].apply(lambda x: group_status(x))

    # minmaxscaling for numerical features age and credit_amount
    # In order to handle the numerical features properly, 'age', 'number_of_credits' and 'credit_amount' are scaled
    # between 0 and 1. To do so, the MinMaxScaler is used.
    df['age'] = scale_min_max('age')
    df['credit_amount'] = scale_min_max('credit_amount')
    df['number_of_credits'] = scale_min_max('number_of_credits')

    return df


