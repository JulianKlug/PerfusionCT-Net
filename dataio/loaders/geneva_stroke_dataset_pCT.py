import torch.utils.data as data
import numpy as np
import datetime
from .utils import validate_images


class GenevaStrokeDataset_pCT(data.Dataset):
    def __init__(self, dataset_path, split, transform=None, preload_data=False):
        '''
        Loader for the Geneva Stroke Dateset (perfusion CT)
        :param dataset_path: path to dataset file
        :param split: split type (train/test/validation)
        :param transform: apply transformations for augmentation
        :param preload_data: boolean, preload data into RAM
        '''
        super(GenevaStrokeDataset_pCT, self).__init__()

        self.dataset_path = dataset_path
        self.params = np.load(dataset_path, allow_pickle=True)['params']
        print('Geneva Stroke Dataset (perfusion CT maps) parameters: ', self.params)

        self.ids = np.load(dataset_path, allow_pickle=True)['ids']

        # report the number of images in the dataset
        print('Number of {0} images: {1}'.format(split, len(self.ids)))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = np.load(dataset_path, allow_pickle=True)['ct_inputs'].astype(np.int16)
            # todo check how to apply mask
            self.raw_masks = np.load(dataset_path, allow_pickle=True)['brain_masks']
            try:
                self.raw_labels = np.load(dataset_path, allow_pickle=True)['ct_lesion_GT'].astype(np.uint8)
            except:
                self.raw_labels = np.load(dataset_path, allow_pickle=True)['lesion_GT'].astype(np.uint8)
            self.raw_labels = np.expand_dims(self.raw_labels, axis=-1)
            assert len(self.raw_images) == len(self.raw_labels)
            print('Loading is done\n')


    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the images
        if not self.preload_data:
            input = np.load(self.dataset_path, allow_pickle=True)['ct_inputs'][index].astype(np.int16)
            try:
                target = np.load(self.dataset_path, allow_pickle=True)['ct_lesion_GT'][index].astype(np.uint8)
            except:
                target = np.load(self.dataset_path, allow_pickle=True)['lesion_GT'][index].astype(np.uint8)
            target = np.expand_dims(target, axis=-1)
            # todo check how to apply mask
            mask = np.load(self.dataset_path, allow_pickle=True)['brain_masks'][index]
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        validate_images(input, target)
        if self.transform:
            input, target = self.transform(input, target)

        return input, target

    def __len__(self):
        return len(self.ids)