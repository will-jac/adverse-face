### adapted to pytorch
### from https://github.com/anishathalye/obfuscated-gradients/baseline/baseline.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class AttackLoss(nn.Module):
    def __init__(self):
        super(AttackLoss, self).__init__()

    def forward(self, f_x_prime, labels):
        # softmax
        f_x_prime = torch.sigmoid(f_x_prime)
        # flatten
        f_x_prime = f_x_prime.view(-1)
        labels = labels.view(-1)
        # loss
        correct_logits = torch.sum(labels*f_x_prime)
        wrong_logits = torch.sum((1 - labels)*f_x_prime - 1e4*labels)

        return torch.sub(correct_logits, wrong_logits)

def attack(x, y, model, num_classes=None, tol=1, num_steps=100, step_size=1, random_start=False, epsilon=8):
    epsilon = 8 / 255.0

    optimizer = torch.optim.Adam(x, lr = step_size*1)
    loss_fn = AttackLoss()

    xprime = x

    y = F.one_hot(y, num_classes)

    for i in range(num_steps):
        optimizer.zero_grad()

        delta = torch.clamp(xprime, 0, 1.0) - x
        delta = torch.clamp(delta, -epsilon, epsilon)

        xprime = x + delta

        logits, pre_softmax = model(xprime)
        print(logits.shape)
        print(pre_softmax.shape)

        # TODO: compute label mask?

        loss = loss_fn(logits, y)
        if i % 100 == 0:
            print('loss:', loss)
        loss.requires_grad = True
        loss.backward()
        optimizer.step()

    return x
