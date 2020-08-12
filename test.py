from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loaders import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.utils import json_file_to_pyobj, save_config
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model
from models.utils import EarlyStopper

def test(json_filename):

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation,
                                              max_output_channels=json_opts.model.output_nc,
                                              verbose=json_opts.training.verbose)

    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, arch_type).scale_size[-1]:
        raise Exception('Number of data channels must match number of model channels, and patch and scale size dimensions')

    # # Setup the NN Model
    # model = get_model(json_opts.model)

    # Setup Data Loader
    split_opts = json_opts.data_split
    train_dataset = ds_class(ds_path, split='train',      transform=ds_transform['train'], preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, channels=channels)
    valid_dataset = ds_class(ds_path, split='validation', transform=ds_transform['valid'], preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, channels=channels)
    test_dataset  = ds_class(ds_path, split='test',       transform=ds_transform['valid'], preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, channels=channels)
    train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=16, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=16, batch_size=train_opts.batchSize, shuffle=False)

    for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
        print(images.shape)
        print(labels.shape)



test('./configs/bayesian_skip/isles2018/test.json')

