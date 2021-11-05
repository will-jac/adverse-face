
from data.datasets import load_data

from attacks.gan.no_box import main as no_box

train_model = False

n_images_per_person = 10

n_iters = 15000
n_decoders = 1
lr = 0.001
train_mode = 'jigsaw'

ce_niters = 200
ila_niters = 100

start_idx = 0
end_idx = 5

if __name__ == '__main__':
        
    data_loader = load_data('lfw', 'train', 
        n_images_per_person, batch_by_people=True
    )

    print('data loaded... training surrogate auto-encoder model')
    no_box(
        data_loader, n_images_per_person,
        train=True,
        n_iters=n_iters, n_decoders=n_decoders, lr=lr, train_mode=train_mode,
        start_idx=start_idx, end_idx=end_idx
    )
    print('done training... generating attack images')
    no_box(
        data_loader, n_images_per_person,
        train=False,
        n_iters=n_iters, n_decoders=n_decoders, lr=lr, train_mode=train_mode,
        start_idx=start_idx, end_idx=end_idx,
        ce_niters=ce_niters, ila_niters=ila_niters
    )


