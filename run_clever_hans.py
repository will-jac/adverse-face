from absl import app, flags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import os
def save_images(
    img_tensor, 
    data_loader, batch_size, data_ind, 
    save_dir='attacks/final_obfuscated_gradient_attack/',
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

    custom = True
    # CUDA_LAUNCH_BLOCKING=1
    batch_size = 10
    # load data
    from data.datasets import load_data, num_classes
    if custom:
        data_loader = load_data('custom', mode='train', batch_size=batch_size, shuffle=True)
        nc = num_classes['lfw'] + 1
    else:
        data_loader = load_data('lfw', True, 'train', 
            batch_size=batch_size, batch_by_people=False, shuffle=True
        )
        nc = num_classes['lfw'] + 1
    # Instantiate model, loss, and optimizer for training
    
    # can use any torch model here
    from attacks.base_models.resnet50_torch import load_resnet_classifier, load_resnet_yny
    if custom:
        model = load_resnet_yny()
    else:
        model = load_resnet_classifier()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device=device)

    save_path = 'attacks/final_obfuscated_gradient_attack/'

    # Evaluate on clean and adversarial data
    model.eval()
    report = {'nb_test':0, 'correct':0, 'correct_fgm':0, 'correct_pgd':0, 'correct_cwg':0}
    
    for x, y in data_loader:
        if batch_size == 1:
            x = x.view(batch_size, 3, 224, 224)
            y = y.view(batch_size, -1) 
        if custom:
            # y = nn.functional.one_hot(torch.as_tensor([nc-1]*len(y)), nc).float()
            y = torch.as_tensor([nc-1]*len(y))

        x, y = x.to(device), y.to(device)
        print(y)
        report['nb_test'] += batch_size

        x_fgm = fast_gradient_method(model, x, eps=0.1, norm=np.inf)
        # lower epx = more like original
        x_pgd = projected_gradient_descent(model, x, eps=0.05, eps_iter=0.0001, nb_iter=2000, norm=np.inf)

        # x_cwg = carlini_wagner_l2(model, x, nc)

        _, y_pred = model(x).max(1)  # model prediction on clean examples
        print(y_pred)
        _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
        print(y_pred_fgm)
        # _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
        # _, y_pred_cwg = model(x_cwg).max(1)

        report['correct'] += y_pred.eq(y).sum().item()
        report['correct_fgm'] += y_pred_fgm.eq(y).sum().item()
        # report['correct_pgd'] += y_pred_pgd.eq(y).sum().item()
        # report['correct_cwg'] += y_pred_cwg.eq(y).sum().item()
        
        torchvision.utils.save_image(x, save_path + "original.png")
        torchvision.utils.save_image(x_fgm, save_path + "fgsm.png")
        # torchvision.utils.save_image(x_pgd, save_path + "pgd.png")
        # torchvision.utils.save_image(x_cwg, save_path + "cwg.png")
        for i in range(batch_size):
            torchvision.utils.save_image(x[i], save_path + "original_"+str(i)+".png")
            torchvision.utils.save_image(x_fgm[i], save_path + "fgsm_"+str(i)+".png")
            # torchvision.utils.save_image(x_pgd[i], save_path + "pgd_"+str(i)+".png")
            # torchvision.utils.save_image(x_cwg[i], save_path + "cwg_"+str(i)+".png")
        
        # do only one batch
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
    # print(
    #     "test acc on PGD adversarial examples (%): {:.3f}".format(
    #         report['correct_pgd'] / report['nb_test'] * 100.0
    #     )
    # )
    # print(
    #     "test acc on CWG adversarial examples (%): {:.3f}".format(
    #         report['correct_cwg'] / report['nb_test'] * 100.0
    #     )
    # )

if __name__ == "__main__":
    flags.DEFINE_integer("nb_epochs", 2, "Number of epochs.")
    flags.DEFINE_float("eps", 0.01, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (on PGD adversarial examples)."
    )

    app.run(main)