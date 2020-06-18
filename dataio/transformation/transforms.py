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
        self.scale_size = (192, 192, 1)  # pad up to this shape
        # Todo implement random crop augmentation
        # self.patch_size = (128, 128, 1)

        # Affine, Elastic, Flip and Noise transformations are forked from torchio
        # Further documentation for all arguments can thus be found here:
        # https://torchio.readthedocs.io/transforms/augmentation.html#augmentation
        # Most values can either be given along every dimension (vx, vy, vz);
        # or as a single value v if the same value should be applied along each dimension


        # Affine and Intensity Transformations
        # self.shift_val = (0.1, 0.1) todo - evaluation relevance of shift augmentation
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.flip_axis = (0)  # axes - Axis or tuple of axes along which the image will be flipped.
        self.flip_prob_per_axis = 0.5
        self.noise_std = (0, 0.25)  # range of noise std
        # self.inten_val = (1.0, 1.0) - Todo implement intensitiy augmentation

        self.max_deform = (7.5, 7.5, 7.5)  # Maximum deformation allowed for elastic transform
        self.elastic_control_points = (7, 7, 7)  # control points along each dimension

        self.random_flip_prob = 0.0
        self.random_affine_prob = 0.0
        self.random_elastic_prob = 0.0
        self.random_noise_prob = 0.0

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts, max_output_channels=10):
        t_opts = getattr(opts, self.name)
        self.max_output_channels = max_output_channels

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):               self.scale_size =           t_opts.scale_size
        # if hasattr(t_opts, 'patch_size'):             self.patch_size =           t_opts.patch_size
        # if hasattr(t_opts, 'shift_val'):              self.shift_val =            t_opts.shift
        if hasattr(t_opts, 'rotate'):                   self.rotate_val =           t_opts.rotate
        if hasattr(t_opts, 'scale_val'):                self.scale_val =            t_opts.scale_val
        if hasattr(t_opts, 'max_deform'):               self.max_deform =           t_opts.max_deform
        if hasattr(t_opts, 'elastic_control_points'):   self.elastic_control_points = t_opts.elastic_control_points
        if hasattr(t_opts, 'flip_axis'):                self.flip_axis =            t_opts.flip_axis
        if hasattr(t_opts, 'flip_prob_per_axis'):       self.flip_prob_per_axis =   t_opts.flip_prob_per_axis
        if hasattr(t_opts, 'noise_std'):                self.noise_std =            t_opts.noise_std
        # if hasattr(t_opts, 'inten_val'):              self.inten_val =            t_opts.intensity
        
        if hasattr(t_opts, 'random_flip_prob'):     self.random_flip_prob =     t_opts.random_flip_prob
        if hasattr(t_opts, 'random_affine_prob'):   self.random_affine_prob =   t_opts.random_affine_prob
        if hasattr(t_opts, 'random_elastic_prob'):  self.random_elastic_prob =  t_opts.random_elastic_prob
        if hasattr(t_opts, 'random_noise_prob'):    self.random_noise_prob =    t_opts.random_noise_prob

    def get_transformation(self):
        '''
        Get transformations for this dataset
        :return:
        '''
        return {
            'gsd_pCT': {'train': self.gsd_pCT_train_transform, 'valid': self.gsd_pCT_valid_transform}
        }[self.name]

    def gsd_pCT_train_transform(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 9999)  # seed must be an integer for torch

        train_transform = ts.Compose([
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.TypeCast(['float', 'float']),
            RandomFlipTransform(axes=self.flip_axis, flip_probability=self.flip_prob_per_axis, p=self.random_flip_prob, seed=seed, max_output_channels=self.max_output_channels),
            RandomElasticTransform(seed=seed, p=self.random_elastic_prob, image_interpolation=Interpolation.BSPLINE, max_displacement=self.max_deform,
                                   num_control_points=self.elastic_control_points,  max_output_channels=self.max_output_channels),
            RandomAffineTransform(scales = self.scale_val, degrees = (self.rotate_val), isotropic = True, default_pad_value = 0,
                        image_interpolation = Interpolation.BSPLINE, seed=seed, p=self.random_affine_prob, max_output_channels=self.max_output_channels),
            RandomNoiseTransform(p=self.random_noise_prob, std=self.noise_std, seed=seed, max_output_channels=self.max_output_channels),
            ts.ChannelsFirst(),
            # ts.NormalizeMedicPercentile(norm_flag=(True, False)),
            # Todo apply channel wise normalisation
            ts.NormalizeMedic(norm_flag=(True, False)),
            # Todo eventually add random crop augmentation (fork torchsample and fix the Random Crop bug)
            # ts.ChannelsLast(), # seems to be needed for crop
            # ts.RandomCrop(size=self.patch_size),
            ts.TypeCast(['float', 'long'])
        ])

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

        return valid_transform
    