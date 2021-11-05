
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import csv
import os
import torch
import torchvision
import numpy as np

supported_datasets = ['lfw']

from sklearn.datasets import fetch_lfw_people # face recognition
# from sklearn.datasets import fetch_lfw_pairs # face verification

# not supported
def load_lfw_sklearn(num_images_per_person):
    dataset = fetch_lfw_people(
        funneled=True,
        min_faces_per_person=num_images_per_person,
        color=True,
        resize=0.4
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_images_per_person, shuffle=False, num_workers = 1
    )

    return data_loader

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

def load_data(dataset_name, mode='train', batch_size=10, shuffle=False, batch_by_people=True):
    dataset_name = dataset_name.lower()
    assert dataset_name in supported_datasets, 'UNRECOGNIZED DATASET, ONLY SUPPORT %s'%(supported_datasets)
    assert mode in ['train', 'attack', 'all'], 'WRONG DATASET MODE, must be in {"train", "attack", "all"}'

    # init dataset and data loader
    if batch_by_people:
        assert batch_size % 2 == 0, 'when batching by people, batch size must be even'
        assert not shuffle, 'does not make sense to shuffle when batch_by_people is True'

    if dataset_name == 'lfw':
        data_loader = load_lfw_torch(batch_size, shuffle, batch_by_people)
        
    return data_loader

