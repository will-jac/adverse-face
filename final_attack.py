import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import tensorflow as tf

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

import PIL
import cv2
from deepface import DeepFace

def preprocess(img_paths, target_sizes=[(224, 224)],
    enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base',
    grayscale=False
):
    # from DeepFace

    if len(target_sizes) == 1:
        imgs = []
    else:
        imgs = [[] for _ in target_sizes]


    for img_path in img_paths:

        img = DeepFace.functions.load_image(img_path)

        base_img = img.copy()

        img, region = DeepFace.functions.detect_face(img = img, 
            detector_backend = detector_backend, grayscale = grayscale, 
            enforce_detection = enforce_detection, align = align
        )

        # #--------------------------

        if img.shape[0] == 0 or img.shape[1] == 0:
            if enforce_detection == True:
                raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
            else: #restore base image
                img = base_img.copy()

        #--------------------------

        #post-processing
        if grayscale == True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #---------------------------------------------------
        #resize image to expected shape 
        base_img = img.copy()

        # img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image
        for i, target_size in enumerate(target_sizes):

            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor)) 
            img = cv2.resize(img, dsize)

            # Then pad the other side to the target size by adding black pixels
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            if grayscale == False:
                # Put the base image in the middle of the padded image
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
            else:
                img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

            #double check: if target image is not still the same size with target.
            if img.shape[0:2] != target_size:
                img = cv2.resize(img, target_size)

            #---------------------------------------------------

            #normalizing the image pixels

            img_pixels = tf.keras.preprocessing.image.img_to_array(img) #what this line doing? must?
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255 #normalize input in [0, 1]

            #---------------------------------------------------

            img_pixels = DeepFace.functions.normalize_input(img = img_pixels, normalization = normalization)

            if len(target_sizes) == 1:
                imgs.append(img_pixels[0])
            else:
                imgs[i].append(img_pixels[0])

            # reset for next iter
            img = base_img.copy()
    
    # print(len(imgs[0]), len(imgs[1]))
    # print(imgs[0][0].shape, imgs[1][0].shape)
    if len(target_sizes) == 1:
        imgs = np.array(imgs)
    else:
        imgs[0] = np.array(imgs[0])
        imgs[1] = np.array(imgs[1])

    return imgs


def main():
    torch.cuda.empty_cache()

    custom = False
    # CUDA_LAUNCH_BLOCKING=1
    batch_size = 1
    total_count = 10
    # load data
    from data.datasets import load_lfw_test

    data_loader, dataset = load_lfw_test(batch_size)

    idx_to_path = {idx : path for (path, idx) in dataset.imgs}

    # Instantiate model, loss, and optimizer for training

    save_path = 'attacks/final_obfuscated_gradient_attack/'

    # Evaluate on clean and adversarial data
    report = {'nb_test':0, 'correct':0, 'correct_fgm':0, 'correct_pgd':0, 'correct_cwg':0}
    
    # get save names
    test_dataset = os.path.join('data', 'lfw-test')
    attack_paths = [None]*len(idx_to_path)
    for i, p in idx_to_path.items():
        n = os.path.split(os.path.split(p)[0])[-1]

        attack_paths[i] = (
            save_path, n, str(i)+'.png'
        )
    print(attack_paths[:10])

    # for subdir in os.listdir(test_dataset):
    #     person_path = os.path.join(test_dataset, subdir)
    #     if os.path.isdir(person_path):
    #         for i, img_file in enumerate(os.listdir(person_path)):
    #             img_path = os.path.join(person_path, img_file)
                
    #             attack_paths.append(
    #                 (
    #                     save_path,
    #                     subdir,
    #                     str(i)+'.png'
    #                 )
    #             )

    # can use any torch model here
    from attacks.base_models.resnet50_torch import load_resnet_classifier, load_resnet_pretrained_yny, load_resnet_yny
    if custom:
        model = load_resnet_yny()
    else:
        model = load_resnet_classifier()
        # model = load_resnet_pretrained_yny()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device=device)

    model.eval()

    i = 0
    for x, y in data_loader:

        # with torch.no_grad():
        #     x_path = original_paths[i:i+batch_size]
        #     # x = np.array([np.asarray(PIL.Image.open(xp), dtype=np.float32) for xp in x_path])
        #     # x = torch.from_numpy(x)
        #     x = preprocess(x_path, enforce_detection=False)
        #     # permute from (10, 224, 224, 3) to (10, 3, 224, 224)
        #     x = np.moveaxis(x, -1, 1)
        #     x = np.array(x, dtype=np.float32)
        
        # x = torch.from_numpy(x)
        x = x.to(device)

        _, y_pred = model(x).max(1)  # model prediction on clean examples
        print(y_pred)

        report['nb_test'] += batch_size

        # x_fgm = fast_gradient_method(model, x, eps=0.1, norm=np.inf)
        # lower epx = more like original
        x_pgd = projected_gradient_descent(model, x, 
            eps=0.01, eps_iter=0.001, nb_iter=500, norm=np.inf
        )
        # x_cwg = carlini_wagner_l2(model, x, nc)

        # _, y_pred_fgm = model(x_fgm).max(1)  # model prediction on FGM adversarial examples
        _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
        print(y_pred_pgd)
        # _, y_pred_pgd = model(x_pgd).max(1)  # model prediction on PGD adversarial examples
        # _, y_pred_cwg = model(x_cwg).max(1)

        # report['correct'] += y_pred.eq(y).sum().item()
        # report['correct_fgm'] += y_pred_fgm.eq(y).sum().item()
        report['correct_pgd'] += y_pred_pgd.eq(y_pred).sum().item()
        # report['correct_cwg'] += y_pred_cwg.eq(y).sum().item()
        
        for j in range(batch_size):
            a_path = os.path.join(attack_paths[i+j][0], attack_paths[i+j][1])
            os.makedirs(a_path, exist_ok=True)
            a_path = os.path.join(a_path, attack_paths[i+j][2])

            torchvision.utils.save_image(x_pgd[j], a_path)
            
        i += 1
        if i > total_count:
            break
        # # torchvision.utils.save_image(x_fgm, save_path + "fgsm.png")
        # torchvision.utils.save_image(x_pgd, save_path + "pgd.png")
        # # torchvision.utils.save_image(x_cwg, save_path + "cwg.png")
        # for i in range(batch_size):
        #     torchvision.utils.save_image(x[i], save_path + "original_"+str(i)+".png")
        #     # torchvision.utils.save_image(x_fgm[i], save_path + "fgsm_"+str(i)+".png")
        #     torchvision.utils.save_image(x_pgd[i], save_path + "pgd_"+str(i)+".png")
        #     # torchvision.utils.save_image(x_cwg[i], save_path + "cwg_"+str(i)+".png")
        
        # do only one batch

    # print(
    #     "test acc on clean examples (%): {:.3f}".format(
    #         report['correct'] / report['nb_test'] * 100.0
    #     )
    # )
    # print(
    #     "test acc on FGM adversarial examples (%): {:.3f}".format(
    #         report['correct_fgm'] / report['nb_test'] * 100.0
    #     )
    # )
    print(
        "test acc on PGD adversarial examples (%): {:.3f}".format(
            report['correct_pgd'] / report['nb_test'] * 100.0
        )
    )
    # print(
    #     "test acc on CWG adversarial examples (%): {:.3f}".format(
    #         report['correct_cwg'] / report['nb_test'] * 100.0
    #     )
    # )

if __name__ == "__main__":
    main()