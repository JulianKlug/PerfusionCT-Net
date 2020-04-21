from dataio.loaders.geneva_stroke_dataset_pCT import GenevaStrokeDataset_pCT

def get_dataset(name):
    """get_dataset

    :param name:
    """
    return {
        'gsd_pCT': GenevaStrokeDataset_pCT
    }[name]


def get_dataset_path(dataset_name, opts):
    """get_data_path

    :param dataset_name:
    :param opts:
    """

    return getattr(opts, dataset_name)
