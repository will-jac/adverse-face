import torch

def difference(img_org, img_atk, metric='dist1'):
    # compute difference using given metric
    if metric == 'dist1':
        score = torch.Tensor.dist(img_org, img_atk, 1) # L1 norm as a single-element Tensor
    elif metric == 'dist2':
        score = torch.Tensor.dist(img_org, img_atk, 2) # L2 norm as a single-element Tensor
    elif metric == 'eq':
        score = torch.Tensor.eq(img_org, img_atk) # equality as a boolean Tensor
    elif metric == 'isclose':
        score = torch.Tensor.isclose(img_org, img_atk) # is_close as a boolean Tensor
    elif metric == 'sub':
        score = torch.Tensor.sub(img_org, img_atk) # difference as a Tensor
    # other possible metrics/useful functions:
    #   Tensor.item() returns the value of the (single-element) tensor as a number
    #   Tensor.numpy() returns the tensor as a numpy ndarray
    return score