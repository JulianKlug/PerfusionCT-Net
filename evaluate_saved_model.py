import os

from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataio.loaders import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from models import get_model
from utils.error_logger import StatLogger
from utils.metrics import dice_score, distance_metric, precision_and_recall, single_class_dice_score, \
    intersection_over_union, specificity
from utils.utils import json_file_to_pyobj, mkdir


def evaluate_saved_model(model_config, split='validation', model_path=None, data_path=None, save_directory=None, save_nii=False, save_npz=False):
    # Load options
    json_opts = json_file_to_pyobj(model_config)
    train_opts = json_opts.training
    model_opts = json_opts.model
    data_path_opts = json_opts.data_path

    if model_path is not None:
        model_opts = json_opts.model._replace(path_pre_trained_model=model_path)

    model_opts = model_opts._replace(gpu_ids=[])

    # Setup the NN Model
    model = get_model(model_opts)
    if save_directory is None:
        save_directory = os.path.join(os.path.dirname(model_config), split + '_evaluation')
    mkdir(save_directory)

    # Setup Dataset and Augmentation
    ds_class = get_dataset(train_opts.arch_type)
    if data_path is None:
        data_path = get_dataset_path(train_opts.arch_type, data_path_opts)
    dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

    # Setup channels
    channels = json_opts.data_opts.channels
    if len(channels) != json_opts.model.input_nc \
            or len(channels) != getattr(json_opts.augmentation, train_opts.arch_type).scale_size[-1]:
        raise Exception('Number of data channels must match number of model channels, and patch and scale size dimensions')


    # Setup Data Loader
    split_opts = json_opts.data_split
    dataset = ds_class(data_path, split=split, transform=dataset_transform['valid'],
                             preload_data=train_opts.preloadData,
                             train_size=split_opts.train_size, test_size=split_opts.test_size,
                             valid_size=split_opts.validation_size, split_seed=split_opts.seed, channels=channels)
    data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=False)

    # Visualisation Parameters
    # visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)

    # Setup stats logger
    stat_logger = StatLogger()

    if save_npz:
        all_predicted = []

    # test
    for iteration, data in tqdm(enumerate(data_loader, 1)):
        model.set_input(data[0], data[1])
        model.test()

        input_arr = np.squeeze(data[0].cpu().numpy()).astype(np.float32)
        prior_arr = np.squeeze(data[0].cpu().numpy())[5].astype(np.int16)
        prior_arr[prior_arr > 0] = 1
        label_arr = np.squeeze(data[1].cpu().numpy()).astype(np.int16)
        ids = dataset.get_ids(data[2])
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)

        # If there is a label image - compute statistics
        dice_vals = dice_score(label_arr, output_arr, n_class=int(2))
        single_class_dice = single_class_dice_score(label_arr, output_arr)
        md, hd = distance_metric(label_arr, output_arr, dx=2.00, k=1)
        precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(2))
        sp = specificity(label_arr, output_arr)
        jaccard = jaccard_score(label_arr.flatten(), output_arr.flatten())

        # compute stats for the prior that is used
        prior_dice = single_class_dice_score(label_arr, prior_arr)
        prior_precision, prior_recall = precision_and_recall(label_arr, prior_arr, n_class=int(2))

        stat_logger.update(split=split, input_dict={'img_name': ids[0],
                                                     'dice_bg': dice_vals[0],
                                                     'dice_les': dice_vals[1],
                                                     'dice2_les': single_class_dice,
                                                     'prec_les': precision[1],
                                                     'reca_les': recall[1],
                                                     'specificity': sp,
                                                     'md_les': md,
                                                     'hd_les': hd,
                                                     'jaccard': jaccard,
                                                     'dice_prior': prior_dice,
                                                     'prec_prior': prior_precision[1],
                                                     'reca_prior': prior_recall[1]
                                                     })

        if save_nii:
            # Write a nifti image
            import SimpleITK as sitk
            input_img = sitk.GetImageFromArray(np.transpose(input_arr[0], (2, 1, 0)));
            input_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
            cbf_img = sitk.GetImageFromArray(np.transpose(input_arr[1], (2, 1, 0)));
            cbf_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
            prior_img = sitk.GetImageFromArray(np.transpose(input_arr[5], (2, 1, 0)));
            prior_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
            label_img = sitk.GetImageFromArray(np.transpose(label_arr, (2, 1, 0)));
            label_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])
            predi_img = sitk.GetImageFromArray(np.transpose(output_arr, (2, 1, 0)));
            predi_img.SetDirection([-1, 0, 0, 0, -1, 0, 0, 0, 1])

            sitk.WriteImage(input_img, os.path.join(save_directory, '{}_img.nii.gz'.format(iteration)))
            sitk.WriteImage(cbf_img, os.path.join(save_directory, '{}_cbf.nii.gz'.format(iteration)))
            sitk.WriteImage(prior_img, os.path.join(save_directory, '{}_prior.nii.gz'.format(iteration)))
            sitk.WriteImage(label_img, os.path.join(save_directory, '{}_lbl.nii.gz'.format(iteration)))
            sitk.WriteImage(predi_img, os.path.join(save_directory, '{}_pred.nii.gz'.format(iteration)))

        if save_npz:
            all_predicted.append(output_arr)


    stat_logger.statlogger2csv(split=split, out_csv_name=os.path.join(save_directory, split + '_stats.csv'))
    for key, (mean_val, std_val) in stat_logger.get_errors(split=split).items():
        print('-', key, ': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val), '-')

    if save_npz:
        np.savez_compressed(os.path.join(save_directory, 'predictions.npz'),
                            predicted=np.array(all_predicted))


# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='CNN Seg Validation Function')
#
#     parser.add_argument('-c', '--config', help='testing config file', required=True)
#     args = parser.parse_args()
#
#     validation(args.config)


# evaluate_saved_model('/Users/julian/temp/BAYESIAN_SKIP_USE_PREDICTED_AS_PRIOR/prediction_of_prior/trained_convolutional_bayesian_skip.json',
#                      model_path='/Users/julian/temp/BAYESIAN_SKIP_USE_PREDICTED_AS_PRIOR/prediction_of_prior/303_net_unet_pct_bayesian_multi_att_dsv.pth',
#                      data_path='/Users/julian/temp/BAYESIAN_SKIP_USE_PREDICTED_AS_PRIOR/prediction_of_prior/rescaled_with_ncct_dataset_with_core.npz',
#                      split='test', save_nii=True, save_npz=True)

evaluate_saved_model('/Users/julian/temp/BAYESIAN_SKIP_MODEL_EVALUATION/noisy_prior/GSD/noisy_prior_convolutional_bayesian_skip/trained_noisy_prior_convolutional_bayesian_skip.json',
                     model_path='/Users/julian/temp/BAYESIAN_SKIP_MODEL_EVALUATION/noisy_prior/GSD/noisy_prior_convolutional_bayesian_skip/223_net_unet_pct_bayesian_multi_att_dsv.pth',
                     data_path='/Users/julian/stroke_datasets/dataset_files/perfusion_data_sets/with_prior/noisy_prior/rescaled_with_ncct_dataset_with_core_noisy5.npz',
                     split='test', save_nii=False, save_npz=False)