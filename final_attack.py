import os

import numpy as np

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import tensorflow as tf

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from data.datasets import load_lfw_test
from attacks.base_models.resnet50_torch import load_resnet_classifier, load_resnet_pretrained_yny, load_resnet_yny
 
custom = False

batch_size = 1
total_count = 10

root_save_path = 'attacks/classifier_proxy/'

def attack(params, surrogate_acc=False):
    torch.cuda.empty_cache()
    
    # load data
    data_loader, dataset = load_lfw_test(batch_size)
    idx_to_path = {idx : path for (path, idx) in dataset.imgs}

    report = {'nb_test':0, 'correct':0, 'correct_pgd':0}

    # get save names
    save_path = root_save_path + '_' + '_'.join([str(params[k]) for k in sorted(list(params.keys()))])

    attack_paths = [None]*len(idx_to_path)
    for i, p in idx_to_path.items():
        n = os.path.split(os.path.split(p)[0])[-1]

        attack_paths[i] = (
            save_path, n, str(i)+'.png'
        )

    ## fix eps_iter
    params['eps_iter'] = params['eps'] * params['eps_iter']

    # can use any torch model here
    model = load_resnet_classifier() # load_resnet_pretrained_yny()

    # send to gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    model = model.to(device=device)

    model.eval()

    i = 0
    for x, y in data_loader:

        x = x.to(device)
        y = y.to(device)

        # attack
        
        x_pgd = projected_gradient_descent(model, x, y=y, **params)
        
        # save images
        with torch.no_grad():
            for j in range(batch_size):
                a_path = os.path.join(attack_paths[i+j][0], attack_paths[i+j][1])
                os.makedirs(a_path, exist_ok=True)
                a_path = os.path.join(a_path, attack_paths[i+j][2])

                torchvision.utils.save_image(x_pgd[j], a_path)

            if surrogate_acc:        
                _, y_pred = model(x).max(1)  # model prediction on clean examples
                _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples

                report['nb_test'] += batch_size
                report['correct_pgd'] += y_pred_pgd.eq(y_pred).sum().item()


        i += 1
        if i > total_count:
            break

        gc.collect()

    if surrogate_acc:
        print(
            save_path, "test acc on PGD adversarial examples (%): {:.3f}".format(
                report['correct_pgd'] / report['nb_test'] * 100.0
            )
        )


if __name__ == "__main__":
    from sklearn.model_selection import ParameterGrid

    param_grid = ParameterGrid(
        {
            'eps' : [1e-3,1e-2,1e-1],
            'eps_iter' : [1e-2,1e-1],
            'norm' : [2,np.inf],
            'nb_iter' : [500],
            'sanity_checks' : [False]
        }
    )
    for i, params in enumerate(param_grid):
        print('%02d/%d'%(i,len(param_grid)), params)
        attack(params)
    