import numpy as np
import torchsample.transforms as ts
# import torchvision.transforms as tv
from torchio.transforms import Interpolation
from .imageTransformations import RandomElasticTransform, RandomAffineTransform, RandomNoiseTransform, RandomFlipTransform
from pprint import pprint


class Transformations:

    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0
        self.random_affine_prob = 0.0
        self.random_elastic_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

        # Maximum allowed for elastic transform
        self.max_deform = (7.5, 7.5, 7.5)

    def get_transformation(self):
        return {
            'gsd_pCT': self.get_gsd_pCT_transformer()
        }[self.name]

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts, max_output_channels=10):
        t_opts = getattr(opts, self.name)
        self.max_output_channels = max_output_channels

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):       self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'):       self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate'):           self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'):        self.scale_val = t_opts.scale_val
        if hasattr(t_opts, 'max_deform'):       self.max_deform = t_opts.max_deform
        if hasattr(t_opts, 'inten_val'):        self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'random_affine_prob'): self.random_affine_prob = t_opts.random_affine_prob
        if hasattr(t_opts, 'random_elastic_prob'): self.random_elastic_prob = t_opts.random_elastic_prob
        if hasattr(t_opts, 'division_factor'):  self.division_factor = t_opts.division_factor

    def get_gsd_pCT_transformer(self):
        return {'train': self.gsd_pCT_train_transform, 'valid': self.gsd_pCT_valid_transform}

    def gsd_pCT_train_transform(self, seed=None):
        if seed is None:
            # seed must be an integer for torch
            seed = np.random.randint(0, 9999)

        train_transform = ts.Compose([
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.TypeCast(['float', 'float']),
            # ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
            # RandomElasticTransform(seed=seed, max_output_channels=self.max_output_channels),
            # RandomFlipTransform(axes=(0), p=self.random_flip_prob, seed=seed, max_output_channels=self.max_output_channels),
            # RandomElasticTransform(seed=seed, p=1, image_interpolation=Interpolation.BSPLINE, max_displacement=self.max_deform,
            #                        max_output_channels=self.max_output_channels),
            RandomAffineTransform(scales = self.scale_val, degrees = (self.rotate_val), isotropic = True, default_pad_value = 0,
                        image_interpolation = Interpolation.BSPLINE, seed=seed, p=self.random_affine_prob, max_output_channels=self.max_output_channels),
            # RandomNoiseTransform(p=0.5, seed=seed, max_output_channels=self.max_output_channels),
            # Todo Random Affine doesn't support channels --> try newer version of torchsample or torchvision
            # ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
            #                 zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
            ts.ChannelsFirst(),
            #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            # Todo apply channel wise normalisation
            ts.NormalizeMedic(norm_flag=(True, False)),
            # Todo fork torchsample and fix the Random Crop bug
            # ts.ChannelsLast(), # seems to be needed for crop
            # ts.RandomCrop(size=self.patch_size),
            ts.TypeCast(['float', 'long'])
        ])

        # train_transform = tv.Compose([
        #     tv.Lambda(lambda a: torch.from_numpy(a)),
        #     ts.Pad(size=self.scale_size),
        #     # tv.Lambda(lambda a: tv.Pad(a, self.get_padding(a))),
        #     # it.PadToScale(scale_size=self.scale_size),
        #     tv.Lambda(lambda a: a.permute(3, 0, 1, 2)),
        #     tv.Lambda(lambda a: a.float()),
        # ])

        return train_transform

    def gsd_pCT_valid_transform(self, seed=None):
        valid_transform = ts.Compose([
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float']),
            # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            ts.NormalizeMedic(norm_flag=(True, False)),
            # ts.ChannelsLast(),
            # ts.SpecialCrop(size=self.patch_size, crop_type=0),
            ts.TypeCast(['float', 'long'])
        ])

        # valid_transform = tv.Compose([
        #     tv.Lambda(lambda a: torch.from_numpy(a)),
        #     ts.Pad(size=self.scale_size),
        #     # tv.Lambda(lambda a: tv.Pad(a, self.get_padding(a))),
        #     # it.PadToScale(scale_size=self.scale_size),
        #     tv.Lambda(lambda a: a.permute(3, 0, 1, 2)),
        #     tv.Lambda(lambda a: a.float()),
        #
        # ])

        return valid_transform

    # def gsd_pCT_transform(self, seed=None):
    #     '''
    #     Data augmentation transformations for the Geneva Stroke dataset (pCT maps)
    #     :return:
    #     '''
    #     if seed is None:
    #         seed = np.random.rand()
    #     print('yoooooooo', seed)
    #
    #     train_transform = ts.Compose([
    #         ts.ToTensor(),
    #         ts.Pad(size=self.scale_size),
    #         ts.TypeCast(['float', 'float']),
    #         # ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
    #         RandomFlipTransform(axes=(0), p=self.random_flip_prob, seed=seed, max_output_channels=self.max_output_channels),
    #         RandomElasticTransform(seed=seed, p=0.5, image_interpolation=Interpolation.BSPLINE, max_displacement=self.max_deform,
    #                                max_output_channels=self.max_output_channels),
    #         RandomAffineTransform(scales = self.scale_val, degrees = (self.rotate_val), isotropic = False, default_pad_value = 0,
    #                     image_interpolation = Interpolation.BSPLINE, seed=seed, p=0.5, max_output_channels=self.max_output_channels),
    #         RandomNoiseTransform(p=0.5, seed=seed, max_output_channels=self.max_output_channels),
    #         # Todo Random Affine doesn't support channels --> try newer version of torchsample or torchvision
    #         # ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
    #         #                 zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
    #         ts.ChannelsFirst(),
    #         #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
    #         # Todo apply channel wise normalisation
    #         ts.NormalizeMedic(norm_flag=(True, False)),
    #         # Todo fork torchsample and fix the Random Crop bug
    #         # ts.ChannelsLast(), # seems to be needed for crop
    #         # ts.RandomCrop(size=self.patch_size),
    #         ts.TypeCast(['float', 'long'])
    #     ])
    #
    #
    #
    #     return {'train': train_transform, 'valid': valid_transform}