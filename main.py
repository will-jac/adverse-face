
from data.datasets import load_data

from attacks.gan.no_box import main as no_box

train_model = True

batch_size = 1

n_iters = 100 #15000
n_decoders = 20
lr = 0.001
train_mode = 'jigsaw'

start_idx = 0
end_idx = 10

data_loader = load_data('lfw', 'train', batch_size)

no_box(
    data_loader, batch_size,
    train_model,
    n_iters, n_decoders, lr, train_mode,
    start_idx=start_idx, end_idx=end_idx
)


