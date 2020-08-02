import numpy as np
import torchsample.transforms as ts
from .imageTransformations import RandomElasticTransform, RandomAffineTransform, RandomNoiseTransform, RandomFlipTransform, StandardizeImage
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
        # Todo verify why shifting and elastic transform destroy image integrity
        self.shift_val = (0, 5)  # translation range
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.flip_axis = (0)  # axes - Axis or tuple of axes along which the image will be flipped.
        self.flip_prob_per_axis = 0.5
        self.noise_std = (0, 0.25)  # range of noise std
        self.noise_mean = 0
        # self.inten_val = (1.0, 1.0) - Todo implement intensity augmentation

        self.max_deform = (7.5, 7.5, 7.5)  # Maximum deformation allowed for elastic transform
        self.elastic_control_points = (7, 7, 7)  # control points along each dimension

        self.random_flip_prob = 0.0
        self.random_affine_prob = 0.0
        self.random_elastic_prob = 0.0
        self.random_noise_prob = 0.0

        self.prudent = True

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts, max_output_channels=10, verbose=True):
        t_opts = getattr(opts, self.name)
        self.max_output_channels = max_output_channels
        self.verbose = verbose

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):               self.scale_size =           t_opts.scale_size
        # if hasattr(t_opts, 'patch_size'):             self.patch_size =           t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):                self.shift_val =            t_opts.shift_val
        if hasattr(t_opts, 'rotate'):                   self.rotate_val =           t_opts.rotate
        if hasattr(t_opts, 'scale_val'):                self.scale_val =            t_opts.scale_val
        if hasattr(t_opts, 'max_deform'):               self.max_deform =           t_opts.max_deform
        if hasattr(t_opts, 'elastic_control_points'):   self.elastic_control_points = t_opts.elastic_control_points
        if hasattr(t_opts, 'flip_axis'):                self.flip_axis =            t_opts.flip_axis
        if hasattr(t_opts, 'flip_prob_per_axis'):       self.flip_prob_per_axis =   t_opts.flip_prob_per_axis
        if hasattr(t_opts, 'noise_std'):                self.noise_std =            t_opts.noise_std
        if hasattr(t_opts, 'noise_mean'):               self.noise_mean =           t_opts.noise_mean
        # if hasattr(t_opts, 'inten_val'):              self.inten_val =            t_opts.intensity
        
        if hasattr(t_opts, 'random_flip_prob'):     self.random_flip_prob =     t_opts.random_flip_prob
        if hasattr(t_opts, 'random_affine_prob'):   self.random_affine_prob =   t_opts.random_affine_prob
        if hasattr(t_opts, 'random_elastic_prob'):  self.random_elastic_prob =  t_opts.random_elastic_prob
        if hasattr(t_opts, 'random_noise_prob'):    self.random_noise_prob =    t_opts.random_noise_prob

        # Define carefullness of transformation (True: do not allow loss of classes due to augmentation / False)
        if hasattr(t_opts, 'prudent'):              self.prudent =              t_opts.prudent

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
            RandomFlipTransform(axes=self.flip_axis, flip_probability=self.flip_prob_per_axis, p=self.random_flip_prob,
                                seed=seed, max_output_channels=self.max_output_channels, prudent=self.prudent),
            RandomElasticTransform(max_displacement=self.max_deform,
                                   num_control_points=self.elastic_control_points,
                                   image_interpolation='bspline',
                                   seed=seed, p=self.random_elastic_prob,
                                   max_output_channels=self.max_output_channels, verbose=self.verbose, prudent=self.prudent),
            RandomAffineTransform(scales=self.scale_val, degrees=self.rotate_val, translation=self.shift_val,
                                  isotropic=True, default_pad_value=0,
                                  image_interpolation='bspline', seed=seed, p=self.random_affine_prob,
                                  max_output_channels=self.max_output_channels, verbose=self.verbose, prudent=self.prudent),
            StandardizeImage(norm_flag=[True, True, True, False]),
            RandomNoiseTransform(mean=self.noise_mean, std=self.noise_std, seed=seed, p=self.random_noise_prob,
                                 max_output_channels=self.max_output_channels, prudent=self.prudent),
            # Todo eventually add random crop augmentation (fork torchsample and fix the Random Crop bug)
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float'])
        ])

        return train_transform

    def gsd_pCT_valid_transform(self, seed=None):
        valid_transform = ts.Compose([
            ts.ToTensor(),
            ts.Pad(size=self.scale_size),
            ts.TypeCast(['float', 'float']),
            StandardizeImage(norm_flag=[True, True, True, False]),
            ts.ChannelsFirst(),
            ts.TypeCast(['float', 'float'])
        ])

        return valid_transform