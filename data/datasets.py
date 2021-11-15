
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import csv
import os
import torch
import torchvision
import numpy as np

supported_datasets = ['lfw']

import tensorflow.keras as k

# for torch -> tensorflow pipeline
class DataGenerator(k.utils.Sequence):
    
    def __init__(self, gen, ncl):
        self.gen = gen
        self.iter = iter(gen)
        self.ncl = ncl

    def __getitem__(self, _):
        try:
            ims, lbs = next(self.iter)
        except StopIteration:
            self.iter = iter(self.gen)
            ims, lbs = next(self.iter)
        ims = np.swapaxes(np.swapaxes(ims.numpy(), 1, 3), 1, 2)
        lbs = np.eye(self.ncl)[lbs].reshape(self.gen.batch_size, self.ncl)
        return ims, lbs

    def __len__(self):
        return len(self.gen)

def load_lfw_torch(batch_size, shuffle, batch_by_people):
    trans = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop(224),
        T.ToTensor()
    ])

    dataset = torchvision.datasets.LFWPeople(
        root='./data', 
        split='train',
        image_set='funneled', 
        transform = trans,
        download=True
    )

    print('max target:', max(dataset.targets))
    if batch_by_people:
        n_img_per_person = batch_size // 2
        # now, permute the dataset so that each batch contains only one person
        data_by_people = {}
        for d, t in zip(dataset.data, dataset.targets):
            if t not in data_by_people:
                data_by_people[t] = []
            if len(data_by_people[t]) < n_img_per_person:
                data_by_people[t].append(d)
        data_2 = []
        targets_2 = []
        for t, d in data_by_people.items():
            if len(d) == n_img_per_person:
                targets_2 += [t]*n_img_per_person
                data_2 += d
            elif len(d) > n_img_per_person:
                assert False, 'error, too many images per person'
        dataset.data = data_2
        dataset.targets = targets_2

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers = 1
    )

    return data_loader

def load_data(dataset_name, torch=True, mode='train', batch_size=10, shuffle=False, batch_by_people=True):
    dataset_name = dataset_name.lower()
    assert dataset_name in supported_datasets, 'UNRECOGNIZED DATASET, ONLY SUPPORT %s'%(supported_datasets)
    assert mode in ['train', 'attack', 'all'], 'WRONG DATASET MODE, must be in {"train", "attack", "all"}'

    # init dataset and data loader
    if batch_by_people:
        assert batch_size % 2 == 0, 'when batching by people, batch size must be even'
        assert not shuffle, 'does not make sense to shuffle when batch_by_people is True'

    if dataset_name == 'lfw':
        data_loader = load_lfw_torch(batch_size, shuffle, batch_by_people)
        if not torch:
            # convert to be suitable for tensorflow
            data_loader = DataGenerator(data_loader, 5749) # technically 5749 classes 

    return data_loader

