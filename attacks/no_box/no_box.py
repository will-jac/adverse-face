### from https://github.com/qizhangli/nobox-attacks

import os
import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.backends import cudnn

from attacks.no_box.utils import *

import torchvision
import torchvision.transforms as T

## TRAIN

def train_unsupervised(
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

# TODO: fix. Need to get the prototype_inds working
def train_prototypical(
    model, optimizer, 
    iter_ind, img,
    batch_size, n_imgs_per_person, 
    n_decoders, n_iters, 
    # prototype_ind_csv_writer,
    do_aug,
):
    # if n_imgs_per_person == 1:
    #     tar_ind_ls = [0, 1]
    # else:
    #     tar_ind_ls = mk_proto_ls(n_imgs_per_person, batch_size)
    # tar_ind_ls = tar_ind_ls[:n_decoders * 2]
    # print(tar_ind_ls.tolist())
    tar_ind_ls = [0, n_imgs_per_person]# + [1]*n_imgs_per_person
    # print(tar_ind_ls)
    # prototype_ind_csv_writer.writerow(tar_ind_ls.tolist())
    img_tar = img[tar_ind_ls]
    if n_decoders != 1:
        img_tar = F.interpolate(img_tar, (56, 56))
    since = time.time()
    for i in range(n_iters):
        rand_ind = torch.cat(
            (
                torch.randint(0, n_imgs_per_person, size=(1,)), 
                torch.randint(n_imgs_per_person, batch_size, size=(1,))
            )
        )
        img_input = img[rand_ind].clone()

        if do_aug:
            img_input = aug(img_input)
        
        assert img_input.shape[3] == 224
        
        outputs, _ = model(img_input)
        gen_img = torch.cat(outputs, dim=0)
        loss = nn.MSELoss()(gen_img, img_tar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(iter_ind + 1, i + 1, round(loss.item(), 5), '{} s'.format(int(time.time() - since)))
    return model

def train_loop(
    device,
    n_decoders, surrogate, lr, n_iters, mode,
    data_loader, batch_size, n_imgs_per_person, save_dir,
    start_idx, end_idx,
    force_retrain
):
    if mode == 'naive':
        permute = naive
    if mode == 'jigsaw':
        permute = jigsaw
    if mode == 'rotate':
        permute = rotate

    for iter_ind, (img, targ) in enumerate(data_loader):
        if not start_idx <= iter_ind < end_idx:
            continue

        model_name = save_dir.split('/')[-2:]
        model_save_dir = os.path.join(save_dir, 'models', '{}.pth'.format(iter_ind))
        if not force_retrain and  os.path.exists(model_save_dir):
            print('Model [{} {}] [{}] already trained, not re-running'.format(
                model_name[0], model_name[1], iter_ind
            ))
            continue
        print('Training model [{} {}] [{}]'.format(
            model_name[0], model_name[1], iter_ind
        ))

        model = initialize_model(surrogate, n_decoders, device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        img = img.to(device)

        if mode == 'prototypical':
            train_prototypical(
                model, optimizer, 
                iter_ind, img,
                batch_size, n_imgs_per_person, 
                n_decoders, n_iters, 
                True
            )
        else:
            model = train_unsupervised(
                model=model, 
                optimizer=optimizer, 
                permute_fun=permute, 
                iter_ind=iter_ind, 
                img=img, 
                n_iters=n_iters
            )

        model.eval()
        torch.save(model.state_dict(), model_save_dir)

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
    attack_niters, eps, alpha, 
    batch_size, n_imgs_per_person,
    ce_method
):
    model.eval()
    ori_img = ori_img.to(device)
    nChannels = 3
    tar_img = []
    for i in range(n_imgs_per_person):
        tar_img.append(ori_img[[i, n_imgs_per_person + i]])
    for i in range(n_imgs_per_person):
        tar_img.append(ori_img[[n_imgs_per_person+i, i]])
    tar_img = torch.cat(tar_img, dim=0)
    tar_img = tar_img.reshape(batch_size,2,nChannels,224,224)
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
        label = torch.tensor([0]*batch_size).long().to(device)
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

# TODO: fix. Need to get the prototype_inds working
def attack_ce_proto(
    model, device, 
    ori_img, 
    attack_niters, eps, alpha, n_decoders, 
    ce_method, 
    batch_size, n_imgs_per_person, 
):
    model.eval()
    ori_img = ori_img.to(device)

    prototype_inds = [0, n_imgs_per_person]# + [1]*n_imgs_per_person
    # print(prototype_inds)
    tar_img = []
    for i in range(n_decoders):
        tar_img.append(ori_img[prototype_inds[0], prototype_inds[1]])
        # tar_img.append(ori_img[[prototype_inds[2*i],prototype_inds[2*i+1]]])
    tar_img = torch.cat(tar_img, dim = 0)

    nChannels = 3
    if n_decoders == 1:
        decoder_size = 224
    else:
        decoder_size = 56
        tar_img = F.interpolate(tar_img, size=(56,56))

    tar_img = tar_img.reshape(n_decoders,2,nChannels,decoder_size,decoder_size).unsqueeze(1)
    tar_img = tar_img.repeat(1,batch_size,1,1,1,1).reshape(batch_size*n_decoders,2,nChannels,decoder_size,decoder_size)
    img = ori_img.clone()

    for i in range(attack_niters):
        if ce_method == 'ifgsm':
            img_x = img
        elif ce_method == 'pgd':
            img_x = img + img.new(img.size()).uniform_(-eps, eps)
        img_x.requires_grad_(True)
        outs, _ = model(img_x)
        outs = torch.cat(outs, dim = 0).unsqueeze(1).repeat(1,2,1,1,1)
        loss_mse_ = nn.MSELoss(reduction='none')(outs,tar_img).sum(dim = (2,3,4))/(nChannels*decoder_size*decoder_size)
        loss_mse = - alpha * loss_mse_
        label = torch.tensor(([0]*n_imgs_per_person+[1]*n_imgs_per_person)*n_decoders).long().to(device)
        loss = nn.CrossEntropyLoss()(loss_mse,label)
        if (i+1) % 50 == 0 or i == 0:
            print('attacking {}, {:0.4f}'.format(i, loss.item()))

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
    n_decoders, surrogate, mode,
    ce_niters, ce_epsilon, ce_alpha, ce_method,
    ila_niters, ila_epsilon,
    data_loader, batch_size, n_imgs_per_person,
    save_dir,
    start_idx, end_idx
):
    for data_ind, (original_img, _) in enumerate(data_loader):
        if not start_idx <= data_ind < end_idx:
            continue
        print('loading model', save_dir.split('/')[-2], data_ind)
        model = initialize_model(surrogate, n_decoders, device)
        model.load_state_dict(torch.load(os.path.join(save_dir, 'models', '{}.pth'.format(data_ind))))
        
        model.eval()
        original_img = original_img.to(device)

        if mode == 'prototypical':
            # prototype_ind_csv = open(ae_dir+'/prototype_ind.csv', 'r')
            # prototype_ind_ls = list(csv.reader(prototype_ind_csv))
            old_att_img = attack_ce_proto(
                model, device, n_decoders=n_decoders, 
                ori_img=original_img, 
                attack_niters=ce_niters, eps=ce_epsilon, alpha=ce_alpha, 
                ce_method=ce_method, 
                batch_size=batch_size, n_imgs_per_person=n_imgs_per_person
            )
        else:
            old_att_img = attack_ce_unsup(
                model, device, original_img, 
                attack_niters = ce_niters, eps = ce_epsilon, alpha=ce_alpha, 
                batch_size=batch_size, n_imgs_per_person = n_imgs_per_person,
                ce_method=ce_method
            )

        att_img = attack_ila(
            model, device, original_img, old_att_img, 
            ila_niters, eps=ila_epsilon
        )
        # TODO: save the image
        for save_ind in range(batch_size):
            # file_path, file_name = dataset.imgs[data_ind * 2*n_imgs + save_ind][0].split('/')[-2:]
            fname = os.path.basename(data_loader.dataset.data[data_ind * batch_size + save_ind])
            # fname = data_loader.dataset.data[data_ind * batch_size + save_ind].split('/')[-1]

            # file_dir = fpath[-3] + '/' + fpath[-2]
            # file_name = fpath[-1]
            img_save_dir = os.path.join(save_dir, 'images', str(data_ind))
            os.makedirs(img_save_dir, exist_ok=True)
            img_save_path = os.path.join(img_save_dir, fname.split('.')[0] + '.png')
            print('saving image at:', img_save_path)
            save_attack_img(
                att_img[save_ind],
                img_save_path
                # os.path.join()
                # img_save_dir + fname.split('.')[0] + '.png'
            )
            print('\r', data_ind * batch_size + save_ind, 'images saved.', end=' ')

def main(
    data_loader,
    batch_size=20,
    train=True,
    force_retrain=False,
    n_iters=15000, 
    n_decoders=1,
    lr=0.001, 
    surrogate='resnet',
    mode='jigsaw', 
    ce_epsilon=0.3,
    ce_niters=200,
    ce_alpha=1.0,
    ce_method='ifgsm',
    ila_epsilon=0.1,
    ila_niters=100,
    start_idx = 0,
    end_idx=2500,
    save_dir='./attacks/no_box', 
    seed=0
):
    surrogate = surrogate.lower()
    assert surrogate in ['resnet', 'vgg', 'resnet_pretrained', 'vgg_pretrained'], 'unrecognized surrogate model!'

    assert batch_size % 2 == 0, 'Batch size must be even'
    n_imgs_per_person = batch_size // 2

    assert n_decoders <= n_imgs_per_person**2, 'Too many decoders'
    # probably should only have one decoder..

    assert mode in ['naive', 'jigsaw', 'rotate', 'prototypical']
    
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        print('running on cuda device')
        device = torch.device('cuda')
    else:
        print('running on cpu')
        device = torch.device('cpu')

    save_dir = '%s/%s_batch_%d_decoders_%d_mode_%s_iters_%d_lr_%s/'%(
        save_dir,
        surrogate, batch_size, n_decoders,
        mode, n_iters, 
        ('%s'%lr).replace('0.','')
    )

    if train:
        os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)

        train_loop(
            device=device, 
            n_decoders=n_decoders, surrogate=surrogate, lr=lr, n_iters=n_iters, mode=mode, 
            data_loader=data_loader, batch_size=batch_size, n_imgs_per_person=n_imgs_per_person, 
            save_dir=save_dir, 
            start_idx=start_idx, end_idx=end_idx,
            force_retrain=force_retrain
        )
    else: # attack
        attack_loop(
            device,
            n_decoders=n_decoders, surrogate=surrogate, mode=mode,
            ce_niters=ce_niters, ce_epsilon=ce_epsilon, ce_alpha=ce_alpha, ce_method=ce_method,
            ila_niters=ila_niters, ila_epsilon=ila_epsilon,
            data_loader=data_loader, batch_size=batch_size, n_imgs_per_person=n_imgs_per_person,
            save_dir=save_dir,
            start_idx=start_idx, end_idx=end_idx
        )
