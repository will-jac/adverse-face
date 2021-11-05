
from data.datasets import load_data

from attacks.no_box.no_box import main as no_box

force_train_model = False

n_images_per_person = 5
batch_size = n_images_per_person * 2

n_iters = 15000
n_decoders = 1
lr = 0.001
mode = 'prototypical'

ce_niters = 200
ila_niters = 100

start_idx = 0
end_idx = 5

surrogate = 'resnet' # no other surrogate supported as of yet

if False: # testing 
    force_train_model = True
    n_iters = 1
    ce_niters = 1
    ila_niters = 1
    start_idx = 6
    end_idx = 7

if __name__ == '__main__':

    data_loader = load_data('lfw', 'train', 
        batch_size=batch_size, batch_by_people=True, shuffle=False
    )
    print('data loaded... training surrogate auto-encoder model')
    no_box(
        data_loader, batch_size,
        train=True, force_retrain=force_train_model,
        n_iters=n_iters, n_decoders=n_decoders, lr=lr, 
        surrogate=surrogate, mode=mode,
        start_idx=start_idx, end_idx=end_idx
    )
    
    print('done training... generating attack images')
    no_box(
        data_loader, batch_size,
        train=False,
        n_iters=n_iters, n_decoders=n_decoders, lr=lr, 
        surrogate=surrogate, mode=mode, 
        start_idx=start_idx, end_idx=end_idx,
        ce_niters=ce_niters, ila_niters=ila_niters
    )


