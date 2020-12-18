from dataio.loaders.geneva_stroke_dataset_pCT import GenevaStrokeDataset_pCT
from dataio.loaders.geneva_stroke_dataset_25D_pCT import GenevaStrokeDataset_25D_pCT
from dataio.loaders.isles2018_training_dataset import Isles2018TrainingDataset

def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'gsd_pCT': GenevaStrokeDataset_pCT,
        'gsd_pCT_25D': GenevaStrokeDataset_25D_pCT,
        'isles2018': Isles2018TrainingDataset,
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
