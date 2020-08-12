from torch.utils.data import DataLoader
from tqdm import tqdm

from dataio.loaders import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.utils import json_file_to_pyobj, save_config
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger

from models import get_model
from models.utils import EarlyStopper

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

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

    # Setup the NN Model
    model = get_model(json_opts.model)
    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

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

    # Visualisation Parameters
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)
    # Setup Early Stopping
    early_stopper = EarlyStopper(json_opts.training.early_stopping, verbose=json_opts.training.verbose)
    for epoch in range(model.which_epoch, train_opts.n_epochs):
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
        train_volumes = []
        validation_volumes = []

        # Training Iterations
        for epoch_iter, (images, labels, indices) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

            ids = train_dataset.get_ids(indices)
            volumes = model.get_current_volumes()
            visualizer.display_current_volumes(volumes, ids, 'train', epoch)
            train_volumes.append(volumes)

        # Validation and Testing Iterations
        for loader, split, dataset in zip([valid_loader, test_loader], ['validation', 'test'], [valid_dataset, test_dataset]):
            for epoch_iter, (images, labels, indices) in tqdm(enumerate(loader, 1), total=len(loader)):
                ids = dataset.get_ids(indices)

                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()

                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)

                if split == 'validation':  # do not look at testing
                    # Visualise predictions
                    volumes = model.get_current_volumes()
                    visualizer.display_current_volumes(volumes, ids, split, epoch)
                    validation_volumes.append(volumes)

                    # Track validation loss values
                    early_stopper.update({**errors, **stats})

        # Update the plots
        for split in ['train', 'validation', 'test']:
            visualizer.plot_current_errors(epoch, error_logger.get_errors(split), split_name=split)
            visualizer.print_current_errors(epoch, error_logger.get_errors(split), split_name=split)
        visualizer.save_plots(epoch, save_frequency=5)
        error_logger.reset()

        # Save the model parameters
        if not early_stopper.is_improving is False:
            model.save(json_opts.model.model_type, epoch)
            save_config(json_opts, json_filename, model, epoch)

        # Update the model learning rate
        model.update_learning_rate(metric=early_stopper.get_current_validation_loss())

        if early_stopper.interrogate(epoch):
            break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Unet Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    train(args)
