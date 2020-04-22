import torch.utils.data as data
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from .utils import validate_images


class GenevaStrokeDataset_pCT(data.Dataset):
    def __init__(self, dataset_path, split, transform=None, preload_data=False,
                 split_seed=42, train_size=0.7, test_size=0.15, valid_size=0.15):
        '''
        Loader for the Geneva Stroke Dateset (perfusion CT)
        :param dataset_path: path to dataset file
        :param split: split type (train/test/validation)
        :param transform: apply transformations for augmentation
        :param preload_data: boolean, preload data into RAM
        '''
        super(GenevaStrokeDataset_pCT, self).__init__()
        # TODO make dataset split

        self.dataset_path = dataset_path
        self.params = np.load(dataset_path, allow_pickle=True)['params']
        print('Geneva Stroke Dataset (perfusion CT maps) parameters: ', self.params)

        self.ids = np.load(dataset_path, allow_pickle=True)['ids']

        dataset_indices = list(range(len(self.ids)))
        test_valid_size = test_size + valid_size
        train_indices, test_val_indices = train_test_split(dataset_indices, train_size=train_size, test_size=test_valid_size,
                                                           random_state=split_seed)
        test_indices, validation_indices = train_test_split(test_val_indices, train_size=test_size/test_valid_size,
                                                             test_size=valid_size/test_valid_size, random_state=split_seed)

        if split == 'train':
            self.split_indices = train_indices
        if split == 'test':
            self.split_indices = test_indices
        if split == 'validation':
            self.split_indices = validation_indices


        self.ids = self.ids[self.split_indices]

        # report the number of images in the dataset
        print('Number of {0} images: {1}'.format(split, len(self.ids)))

        # data augmentation
        self.transform = transform

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = np.load(dataset_path, allow_pickle=True)['ct_inputs'][self.split_indices][..., 0].astype(np.int16)
            # todo check how to apply mask
            self.raw_masks = np.load(dataset_path, allow_pickle=True)['brain_masks'][self.split_indices]
            try:
                self.raw_labels = np.load(dataset_path, allow_pickle=True)['ct_lesion_GT'][self.split_indices].astype(np.uint8)
            except:
                self.raw_labels = np.load(dataset_path, allow_pickle=True)['lesion_GT'][self.split_indices].astype(np.uint8)
            # self.raw_labels = np.expand_dims(self.raw_labels, axis=-1)
            assert len(self.raw_images) == len(self.raw_labels)
            print('Loading is done\n')

    def get_ids(self, indices):
        return [self.ids[index] for index in indices]

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the images
        if not self.preload_data:
            split_specific_index = self.split_indices[index]
            input = np.load(self.dataset_path, allow_pickle=True)['ct_inputs'][split_specific_index, ..., 0].astype(np.int16)
            try:
                target = np.load(self.dataset_path, allow_pickle=True)['ct_lesion_GT'][split_specific_index].astype(np.uint8)
            except:
                target = np.load(self.dataset_path, allow_pickle=True)['lesion_GT'][split_specific_index].astype(np.uint8)
            # target = np.expand_dims(target, axis=-1)
            # todo check how to apply mask
            mask = np.load(self.dataset_path, allow_pickle=True)['brain_masks'][split_specific_index]
        else:
            # With preload, it is already only the images from a certain split that are loaded
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        # handle exceptions
        validate_images(input, target)
        if self.transform:
            input, target = self.transform(input, target)

        return input, target, index

    def __len__(self):
        return len(self.ids)