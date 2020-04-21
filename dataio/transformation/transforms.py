import torchsample.transforms as ts
from pprint import pprint


class Transformations:

    def __init__(self, name):
        self.name = name

        # Input patch and scale size
        self.scale_size = (192, 192, 1)
        self.patch_size = (128, 128, 1)
        # self.patch_size = (208, 272, 1)

        # Affine and Intensity Transformations
        self.shift_val = (0.1, 0.1)
        self.rotate_val = 15.0
        self.scale_val = (0.7, 1.3)
        self.inten_val = (1.0, 1.0)
        self.random_flip_prob = 0.0

        # Divisibility factor for testing
        self.division_factor = (16, 16, 1)

    def get_transformation(self):
        return {
            'gsd_pCT': self.gsd_pCT_transform
        }[self.name]()

    def print(self):
        print('\n\n############# Augmentation Parameters #############')
        pprint(vars(self))
        print('###################################################\n\n')

    def initialise(self, opts):
        t_opts = getattr(opts, self.name)

        # Affine and Intensity Transformations
        if hasattr(t_opts, 'scale_size'):       self.scale_size = t_opts.scale_size
        if hasattr(t_opts, 'patch_size'):       self.patch_size = t_opts.patch_size
        if hasattr(t_opts, 'shift_val'):        self.shift_val = t_opts.shift
        if hasattr(t_opts, 'rotate_val'):       self.rotate_val = t_opts.rotate
        if hasattr(t_opts, 'scale_val'):        self.scale_val = t_opts.scale
        if hasattr(t_opts, 'inten_val'):        self.inten_val = t_opts.intensity
        if hasattr(t_opts, 'random_flip_prob'): self.random_flip_prob = t_opts.random_flip_prob
        if hasattr(t_opts, 'division_factor'):  self.division_factor = t_opts.division_factor

    def gsd_pCT_transform(self):
        '''
        Data augmentation transformations for the Geneva Stroke dataset (pCT maps)
        :return:
        '''

        train_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      ts.RandomFlip(h=True, v=True, p=self.random_flip_prob),
                                      ts.RandomAffine(rotation_range=self.rotate_val, translation_range=self.shift_val,
                                                      zoom_range=self.scale_val, interp=('bilinear', 'nearest')),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.RandomCrop(size=self.patch_size),
                                      ts.TypeCast(['float', 'long'])
                                ])

        valid_transform = ts.Compose([ts.PadNumpy(size=self.scale_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      #ts.NormalizeMedicPercentile(norm_flag=(True, False)),
                                      ts.NormalizeMedic(norm_flag=(True, False)),
                                      ts.ChannelsLast(),
                                      ts.AddChannel(axis=0),
                                      ts.SpecialCrop(size=self.patch_size, crop_type=0),
                                      ts.TypeCast(['float', 'long'])
                                ])

        return {'train': train_transform, 'valid': valid_transform}