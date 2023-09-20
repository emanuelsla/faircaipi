import aif360.datasets
from aif360.algorithms.preprocessing import Reweighing


def reweigh_dataset(data: aif360.datasets.Dataset, unprivileged_groups: dict, privileged_groups: dict):
    """
    Reweighs dataset using the AIF360 Reweighing Method, so that Statistical Parity should be increased/decreased to
    an optimum of 0

    :param data: AIF StandardDataset type
    :type data: aif360.datasets.Dataset
    :param unprivileged_groups: Mapping of protected group to integers
    :type unprivileged_groups: dict
    :param privileged_groups: Mapping of privileged group to integers
    :type privileged_groups: dict
    :return: reweighted StandardDataset
    """
    reweighing = Reweighing(unprivileged_groups=[unprivileged_groups],
                            privileged_groups=[privileged_groups])
    reweighing.fit(data)
    data_reweighed = reweighing.transform(data)

    return data_reweighed
