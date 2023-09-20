import aif360.datasets
import numpy


def init_dataset(aif_dataset, protected, privileged_class: numpy.ndarray, categorical_features, features_to_keep):
    """
    Initializes an AIF built-in Dataset (German, Adult, COMPAS, Bank ...) and transforms it into a StandardDataset type.
    see https://aif360.readthedocs.io/en/stable/modules/datasets.html#common-datasets

    :param aif_dataset: AIF built-in dataset
    :type aif_dataset: aif360.datasets.Dataset
    :param protected: selected binary protected attribute, e.g. 'sex', 'age', 'race' or other
    :type s: str
    :param privileged_class: privileged class of protected attribute
    :type privileged_class: numpy.ndarray
    :param categorical_features: Features with categorical values
    :type categorical_features: numpy.ndarray
    :param features_to_keep: Features that will not be dropped after initialization of the dataset
    :type features_to_keep: numpy.ndarray
    :return: aif360.datasets.Dataset
    """

    dataset = aif_dataset(protected_attribute_names=protected,
                          privileged_classes=privileged_class, categorical_features=categorical_features,
                          features_to_keep=features_to_keep)
    return dataset


def dataset_as_dataframe(dataset: aif360.datasets.Dataset):
    """
    Converts Dataset type StandardDataset into pandas DataFrame type

    :param dataset: StandardDataset
    :return: pandas.DataFrame
    """
    dataframe = dataset.convert_to_dataframe()[0]
    return dataframe


def dataset_as_dict(dataset: aif360.datasets.Dataset):
    """
    Generates dictionary for feature names and description of the dataset

    :param dataset: StandardDataset
    :type dataset: aif360.datasets.Dataset
    :return: dict
    """
    dictionary = dataset.convert_to_dataframe()[1]
    return dictionary


def split_data_into_X_and_y(dataset: aif360.datasets.Dataset):
    """
    Splits StandardDataset into variables X and labels y

    :param dataset: StandardDataset
    :type dataset: aif360.datasets.Dataset
    :return: two Arrays with type numpy.ndarray
    """
    X = dataset.features
    y = dataset.labels.ravel()
    return X, y

