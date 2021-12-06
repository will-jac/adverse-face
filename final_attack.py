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

import sys
import shutil

custom = False

batch_size = 20
start = 0
end = 50
# start_at = int(sys.argv[1])
# total_count = 10

root_save_path = 'attacks/classifier_proxy/'

def attack(params, surrogate_acc=False):
    
    torch.cuda.empty_cache()
    cpu = torch.device('cpu')
    gpu = torch.device('cuda')
    device = gpu if torch.cuda.is_available() else cpu
    
    # load data
    data_loader, dataset = load_lfw_test(batch_size)

    report = {'nb_test':0, 'correct':0, 'correct_pgd':0}

    # get save names
    save_path = root_save_path + 'pgd_' + '_'.join([str(params[k]) for k in sorted(list(params.keys()))])
    class_idx_to_path = {idx : path for (path, idx) in dataset.imgs}
    idx_to_path = {}

    attack_path = 'attacks/test/'
    
    ## fix eps_iter
    params['eps_iter'] = params['eps'] * params['eps_iter']

    # can use any torch model here
    model = load_resnet_classifier() # load_resnet_pretrained_yny()
    model = model.to(device)
    model.eval()
    # # send to gpu

    # # device = torch.device('cpu')

    # torch.cuda.empty_cache()

    # model = model.to(device=device)

    i = start
    for idx, (x, y) in enumerate(data_loader):
        if idx*batch_size < i:
            continue

        for j in range(batch_size):
            idx_to_path[i + j] = class_idx_to_path[y[j].detach().item()]

        print(torch.cuda.memory_allocated())
        
        x, y = x.to(device), y.to(device)

        # attack
        x_pgd = projected_gradient_descent(model, x, y=y, **params)
            
        # save images (on the cpu)
        # x_pgd = x_pgd.to(cpu)

        with torch.no_grad():
            # torchvision.utils.save_image(x_pgd, attack_path + '_'+str(idx)+'.png')
            for j in range(batch_size):
                torchvision.utils.save_image(
                    x_pgd[j], 
                    attack_path + str(i+j)+'.png'
                )
                # torchvision.utils.save_image(x_pgd[j], attack_paths[i+j])

            # if surrogate_acc:        
            #     _, y_pred = model(x).max(1)  # model prediction on clean examples
            #     _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples

            #     report['nb_test'] += batch_size
            #     report['correct_pgd'] += y_pred_pgd.eq(y_pred).sum().item()

        # del x, y, x_pgd
            
            i += batch_size
            if i > end:
                break

        torch.cuda.empty_cache()

        gc.collect()

    # move to the correct location
    attack_paths = [None]*(end - start)

    for i in range(start, end):
        p = idx_to_path[i]
        n = os.path.split(os.path.split(p)[0])[-1]
        ap = os.path.join(save_path, n)
        os.makedirs(ap, exist_ok=True)
        
        attack_paths[i-start] = os.path.join(ap,str(i)+'.png')

    print(attack_paths[i-start])

    for i in range(start, end):
        shutil.move(
            attack_path + str(i)+'.png', 
            attack_paths[i-start]
        )

    if surrogate_acc:
        print(
            save_path, "test acc on PGD adversarial examples (%): {:.3f}".format(
                report['correct_pgd'] / report['nb_test'] * 100.0
            )
        )
    
    return save_path

from deepface_surrogate_attack import eval_attack

if __name__ == "__main__":

    # from sklearn.model_selection import ParameterGrid

    # param_grid = ParameterGrid(
    #     {
    #         'eps' : [1e-3,1e-2,1e-1],
    #         'eps_iter' : [1e-2,1e-1],
    #         'norm' : [np.inf],
    #         'nb_iter' : [500],
    #         'sanity_checks' : [False]
    #     }
    # )
    # for i, params in enumerate(param_grid):
    #     print('%02d/%d'%(i,len(param_grid)), params)
    #     save_path = attack(params)
    #     eval_attack(save_path)

    save_path = attack({
        'eps': 0.09, 'eps_iter':0.005, 'nb_iter':1000, 'norm':np.inf, 'sanity_checks':False
    })
    eval_attack(save_path)