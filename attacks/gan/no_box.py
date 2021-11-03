### from https://github.com/qizhangli/nobox-attacks

import os
import sys
import time
import csv
import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F

from attacks.gan.model_autoencoder import *
from attacks.gan.utils import *

import torchvision

def save_attack_img(img_tensor, file_path):
    # T.ToPILImage()(img.data.cpu()).save(file_dir)
    torchvision.utils.save_image(img_tensor, file_path)

class ILA(torch.nn.Module):
    def __init__(self):
        super(ILA, self).__init__()
    def forward(self, ori_mid, tar_mid, att_mid):
        bs = ori_mid.shape[0]
        ori_mid = ori_mid.view(bs, -1)
        tar_mid = tar_mid.view(bs, -1)
        att_mid = att_mid.view(bs, -1)
        W = att_mid - ori_mid
        V = tar_mid - ori_mid
        V = V / V.norm(p=2,dim=1, keepdim=True)
        ila = (W*V).sum() / bs
        return ila

def initialize_model(decoder_num, device):
    model = autoencoder(
        input_nc=3, output_nc=3, n_blocks=3, decoder_num=decoder_num
    )
    model = nn.Sequential(
        Normalize(),
        model,
    )
    model.to(device)
    return model


## TRAIN
def naive():
    pass
def jigsaw(img_input, img_ind):
    img_input[img_ind] = shuffle(img_input[img_ind], 1)
def rotate(img_input, img_ind):
    img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])

def train_unsup(
    model, optimizer, permute_fun,
    iter_ind, img,  
    n_iters
):
    img_input = img
    img_tar = img.clone()
    since = time.time()
    # mini-batch - img is a single batch, we're fitting to it
    for i in range(n_iters):
        for img_ind in range(img_input.shape[0]):
            permute_fun(img_input, img_ind)
            # if args.mode == 'rotate':
            #     img_input[img_ind:img_ind + 1] = rot(img_input[img_ind:img_ind + 1])
            # elif args.mode == 'jigsaw':
            #     img_input[img_ind] = shuffle(img_input[img_ind], 1)
        outputs, _ = model(img_input)
        loss = nn.MSELoss()(outputs[0], img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
    return model

def train_loop(
    device,
    n_decoders, lr, n_iters, mode,
    data_loader, save_dir,
    start_idx, end_idx
):
    if mode == 'unsup_naive':
        permute = naive
    if mode == 'jigsaw':
        permute = jigsaw
    if mode == 'rotate':
        permute = rotate

    for iter_ind, (img, _) in enumerate(data_loader):
        if not start_idx <= iter_ind < end_idx:
            continue
        model = initialize_model(n_decoders, device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        img = img.to(device)

        train_unsup(model, optimizer, permute, iter_ind, img, n_iters)

        model.eval()
        torch.save(model.state_dict(), save_dir + '/models/{}.pth'.format(iter_ind))

## ATTACK
def attack_ila(model, device, ori_img, tar_img, attack_niters, eps):
    model.eval()
    ori_img = ori_img.to(device)
    img = ori_img.clone()
    with torch.no_grad():
        _, tar_h_feats = model(tar_img)
        _, ori_h_feats = model(ori_img)
    for i in range(attack_niters):
        img.requires_grad_(True)
        _, att_h_feats = model(img)
        loss = ILA()(ori_h_feats.detach(), tar_h_feats.detach(), att_h_feats)
        if (i+1) % 50 == 0:
            print('\r ila attacking {}, {:0.4f}'.format(i+1, loss.item()),end=' ')
        loss.backward()
        input_grad = img.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data

def attack_ce_unsup(
    model, device, 
    ori_img, 
    attack_niters, eps, alpha, n_imgs, ce_method
):
    model.eval()
    ori_img = ori_img.to(device)
    nChannels = 3
    tar_img = []
    for i in range(n_imgs):
        tar_img.append(ori_img[[i, n_imgs + i]])
    for i in range(n_imgs):
        tar_img.append(ori_img[[n_imgs+i, i]])
    tar_img = torch.cat(tar_img, dim=0)
    tar_img = tar_img.reshape(2*n_imgs,2,nChannels,224,224)
    img = ori_img.clone()

    if ce_method == 'pgd':
        # In our implementation of PGD, we incorporate randomness at each iteration to further enhance the transferability
        method = lambda img : img + img.new(img.size()).uniform_(-eps, eps)
    elif ce_method == 'ifgsm':
        method = lambda img : img

    for i in range(attack_niters):
        img_x = method(img)
        img_x.requires_grad_(True)
        outs, _ = model(img_x)
        outs = outs[0].unsqueeze(1).repeat(1, 2, 1, 1, 1)
        loss_mse_ = nn.MSELoss(reduction='none')(outs, tar_img).sum(dim = (2,3,4)) / (nChannels*224*224)
        loss_mse = - alpha * loss_mse_
        label = torch.tensor([0]*n_imgs*2).long().to(device)
        loss = nn.CrossEntropyLoss()(loss_mse,label)
        if (i+1) % 50 == 0 or i == 0:
            print('\r attacking {}, {:0.4f}'.format(i, loss.item()), end=' ')
        loss.backward()
        input_grad = img_x.grad.data.sign()
        img = img.data + 1. / 255 * input_grad
        img = torch.where(img > ori_img + eps, ori_img + eps, img)
        img = torch.where(img < ori_img - eps, ori_img - eps, img)
        img = torch.clamp(img, min=0, max=1)
    print('')
    return img.data

def attack_loop(
    device, 
    n_decoders, model_dir,
    ce_niters, ce_epsilon, ce_alpha, ce_method,
    ila_niters, ila_epsilon,
    dataloader, batch_size,
    start_idx, end_idx
):
    for data_ind, (original_img, _) in enumerate(dataloader):
        if not start_idx <= data_ind < end_idx:
            continue
        model = initialize_model(n_decoders)
        model.load_state_dict(torch.load('{}/models/{}.pth'.format(model_dir, data_ind)))
        model.eval()
        original_img = original_img.to(device)

        old_att_img = attack_ce_unsup(
            model, device, original_img, 
            attack_niters = ce_niters,
            eps = ce_epsilon, alpha=ce_alpha, n_imgs=batch_size,
            ce_method=ce_method
        )

        att_img = attack_ila(
            model, device, original_img, old_att_img, 
            ila_niters, eps=ila_epsilon
        )
        # TODO: save the image
        # for save_ind in range(batch_size):
        #     file_path, file_name = dataset.imgs[data_ind * 2*n_imgs + save_ind][0].split('/')[-2:]
        #     os.makedirs(save_dir + '/' + file_path, exist_ok=True)
        #     save_attack_img(img=att_img[save_ind],
        #                     file_dir=os.path.join(save_dir, file_path, file_name[:-5]) + '.png')
        #     print('\r', data_ind * batch_size + save_ind, 'images saved.', end=' ')

def main(
    data_loader,
    batch_size,
    train=True,
    n_iters=15000, 
    n_decoders=20,
    lr=0.001, 
    train_mode='jigsaw', 
    model_save_dir='./gan/trained_ae', 
    ce_epsilon=0.3,
    ce_niters=200,
    ce_alpha=1.0,
    ce_method='ifgsm',
    ila_epsilon=0.1,
    ila_niters=100,
    img_save_dir='./adv_images',
    start_idx = 0,
    end_idx=2500,
    seed=0
):
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_save_dir = '%s/%s_%s_%s_%s/'%(model_save_dir,train_mode,n_iters,n_decoders,lr)

    if train:
        # prototypical not supported for FR
        assert train_mode in ['unsup_naive', 'jigsaw', 'rotate']
        n_decoders = 1
    
        os.makedirs(model_save_dir + '/models/', exist_ok=True)

        train_loop(
            device, 
            n_decoders, lr, n_iters, 
            train_mode, 
            data_loader, model_save_dir, start_idx, end_idx
        )
    else: # attack
        attack_loop(
            device,
            n_decoders, model_save_dir,
            ce_niters, ce_epsilon, ce_alpha, ce_method,
            ila_niters, ila_epsilon,
            data_loader, batch_size,
            start_idx, end_idx
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--n_iters', type=int, default=15000)
    parser.add_argument('--n_decoders', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mode', type=str, default='jigsaw')
    parser.add_argument('--save_dir', type=str, default='./trained_ae')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=250)

    args = parser.parse_args()
    print(args)

    from data.datasets import load_data

    data_loader = load_data('lfw', 'train', 20)
  
    main(
        data_loader,
        args.n_iters, args.n_decoders, args.lr, args.mode, 
        args.save_dir,
        start_idx = args.start, end_idx = args.end
    )
