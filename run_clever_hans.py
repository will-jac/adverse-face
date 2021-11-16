from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

import os
def save_images(
    img_tensor, 
    data_loader, batch_size, data_ind, 
    save_dir='attacks/obfuscated_gradient/images',
    d = 'fgsm'
):
    for save_ind in range(batch_size):
        fname = os.path.basename(data_loader.dataset.data[data_ind * batch_size + save_ind])

        img_save_dir = os.path.join(save_dir, d)
        os.makedirs(img_save_dir, exist_ok=True)
        img_save_path = os.path.join(img_save_dir, fname.split('.')[0] + '.png')
        # print('saving image at:', img_save_path)
        torchvision.utils.save_image(img_tensor, img_save_path)

FLAGS = flags.FLAGS

def main(_):

    batch_size = 16
    # load data
    from data.datasets import load_data
    data_loader = load_data('lfw', True, 'train', 
        batch_size=batch_size, batch_by_people=False, shuffle=True
    )

    # Instantiate model, loss, and optimizer for training
    
    # can use any torch model here
    from attacks.base_models.resnet50_torch import load_classifier
    model = load_classifier()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()

    # Evaluate on clean and adversarial data
    model.eval()
    report = {'nb_test':0, 'correct':0, 'correct_fgm':0, 'correct_pgd':0}
    for x, y in data_loader:
        x = x.view(batch_size, 3, 224, 224)
        y = y.view(batch_size, -1) 

        x, y = x.to(device), y.to(device)

        x_fgm = fast_gradient_method(model, x, eps=0.1, norm=np.inf)
        # higher nb_iter = more like original
        x_pgd = projected_gradient_descent(model, x, eps=0.01, eps_iter=0.0001, nb_iter=2000, norm=np.inf)

        _, y_pred = model(x).max(1)  # model prediction on clean examples
        _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
        _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples

        # print(y)
        # print(y_pred, y_pred_fgm, y_pred_pgd)
        report['nb_test'] += y.size(0)
        report['correct'] += y_pred.eq(y).sum().item()
        report['correct_fgm'] += y_pred_fgm.eq(y).sum().item()
        report['correct_pgd'] += y_pred_pgd.eq(y).sum().item()
        
        torchvision.utils.save_image(x_fgm, "fgsm.png")
        torchvision.utils.save_image(x_pgd, "pgd.png")
        
        # do only one example
        break
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report['correct'] / report['nb_test'] * 100.0
        )
    )
    print(
        "test acc on FGM adversarial examples (%): {:.3f}".format(
            report['correct_fgm'] / report['nb_test'] * 100.0
        )
    )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report['correct_pgd'] / report['nb_test'] * 100.0
        )
    )


if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 2, "Number of epochs.")
    flags.DEFINE_float("eps", 0.01, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)