
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import csv
import os
import torch
import torchvision
import numpy as np

supported_datasets = ['lfw']

# from sklearn.datasets import fetch_lfw_people # face recognition
# from sklearn.datasets import fetch_lfw_pairs # face verification

def load_lfw(batch_size):
    # lfw_people = fetch_lfw_people(min_faces_per_person=num_images, resize=0.4)
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

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers = 1
    )

    return data_loader

def load_data(dataset_name, mode='train', batch_size=64):
    dataset_name = dataset_name.lower()
    assert dataset_name in supported_datasets, 'UNRECOGNIZED DATASET, ONLY SUPPORT %s'%(supported_datasets)
    assert mode in ['train', 'attack', 'all'], 'WRONG DATASET MODE, must be in {"train", "attack", "all"}'

    # init dataset and data loader

    if dataset_name == 'lfw':
        data_loader = load_lfw(batch_size)
        

    return data_loader

