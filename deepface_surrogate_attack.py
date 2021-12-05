import os
from os import path
import time

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import pairwise_distances 

import torchvision

import tensorflow as tf

from deepface import DeepFace

from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.tf2.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent


def save_images(
    img_tensor,
    data_loader, batch_size, data_ind,
    save_dir='attacks/obfuscated_gradient/images',
    d='fgsm'
):
    for save_ind in range(batch_size):
        fname = os.path.basename(
            data_loader.dataset.data[data_ind * batch_size + save_ind])

        img_save_dir = os.path.join(save_dir, d)
        os.makedirs(img_save_dir, exist_ok=True)
        img_save_path = os.path.join(
            img_save_dir, fname.split('.')[0] + '.png')
        # print('saving image at:', img_save_path)
        torchvision.utils.save_image(img_tensor, img_save_path)


def init_deepface(data_dir_path, models, distance_metric):
    for m in models:
        # print('finding:', os.path.join(data_dir_path, 'lfw-py', 'lfw_funneled',
        #                           'Aaron_Eckhart', 'Aaron_Eckhart_0001.jpg'))
        DeepFace.find(
            img_path=os.path.join('data', 'lfw-py', 'lfw_funneled','Aaron_Eckhart', 'Aaron_Eckhart_0001.jpg'),
            db_path=data_dir_path,
            model_name=m,
            enforce_detection=False
        )

def get_model(model_name):
    # options:
    # VGG-Face, Google FaceNet, OpenFace, Facebook DeepFace, DeepID, ArcFace, Dlib.
    return DeepFace.build_model(model_name)


def preprocess(img_paths, target_sizes=[(224, 224)],
    enforce_detection = True, detector_backend = 'opencv', align = True, normalization = 'base',
    grayscale=False
):
    # from DeepFace

    if len(target_sizes) == 0:
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

        #--------------------------

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


def load_representations(db_path, model_name='VGG-Face'):
    # from DeepFace
    # modified to remove some printing / warnings and speed up computation

    file_name = "representations_%s.pkl" % (model_name)
    file_name = file_name.replace("-", "_").lower()

    if path.exists(db_path+"/"+file_name):

        f = open(db_path+'/'+file_name, 'rb')
        representations = pickle.load(f)

        print("There are", len(representations),
              "representations found in", db_path+'/'+file_name)

        return representations
    else:
        print("WARNING: Representations for images in ", db_path,
              " folder do not exist! Please init deepface.")
        exit(-1)

def find(
    preprocessed_images, representation,
    model, model_name='VGG-Face',
    distance_metric='cosine', 
    enforce_detection=True, detector_backend='opencv', align=True, prog_bar=True, normalization='base'
):
    # from DeepFace
    # modified to remove some printing / warnings and speed up computation 
    # and to fix an error when running find on a batch

    if model == None:
        print('must pass in model!')
        exit(-1)

    columns = ['identity', '%s_representation'%model_name]

    df = pd.DataFrame(representation, columns = columns)

    #df will be filtered in each img. we will restore it for the next item.
    df_base = df.copy()

    resp_obj = []

    threshold = DeepFace.dst.findThreshold(model_name, distance_metric)

    target_representation = [t.tolist() for t in model.predict(preprocessed_images)]

    for i in range(len(preprocessed_images)):

        # find representation for passed image

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance["%s_representation" % model_name]

            if distance_metric == 'cosine':
                distance = DeepFace.dst.findCosineDistance(
                    source_representation, target_representation[i]
                )
            elif distance_metric == 'euclidean':
                distance = DeepFace.dst.findEuclideanDistance(
                    source_representation, target_representation[i]
                )
            elif distance_metric == 'euclidean_l2':
                distance = DeepFace.dst.findEuclideanDistance(
                    DeepFace.dst.l2_normalize(source_representation), 
                    DeepFace.dst.l2_normalize(target_representation[i])
                )

            distances.append(distance)

        
        df["%s_%s" % (model_name, distance_metric)] = distances

        df = df.drop(columns = ["%s_representation" % model_name])
        df = df[df["%s_%s" % (model_name, distance_metric)] <= threshold]

        df = df.sort_values(by = ["%s_%s" % (model_name, distance_metric)], ascending=True).reset_index(drop=True)

        resp = (df['identity'][0], df["%s_%s" % (model_name, distance_metric)][0])

        resp_obj.append(resp)
        df = df_base.copy() #restore df for the next iteration

    return resp_obj

# global results storage
results = {}
original_paths = []

batch_size = 10

distance_metric = 'l2'

model_name = 'VGG-Face'
eval_model_name = 'Facenet512'

def attack(attack_fun, params, idx_start = 0, num_images = 10, save_path=None):

    m_name = eval_model_name 
    
    print('getting model',m_name)
    model = get_model(m_name)
    
    input_shapes = (
        DeepFace.functions.find_input_shape(model), DeepFace.functions.find_input_shape(eval_model)
    )
    
    if save_path is None:
        param_str = attack_fun.__name__ + '_' + '_'.join([str(params[k]) for k in sorted(list(params.keys()))])
        save_path = os.path.join('data', param_str)

    os.makedirs(save_path, exist_ok=True)
    
    attack_paths = []

    # run on the test dataset
    test_dataset = os.path.join('data', 'lfw-test')

    for subdir in os.listdir(test_dataset):
        person_path = os.path.join(test_dataset, subdir)
        if os.path.isdir(person_path):
            for i, img_file in enumerate(os.listdir(person_path)):
                img_path = os.path.join(person_path, img_file)

                original_paths.append(img_path)    
                
                attack_paths.append(
                    (
                        save_path,
                        subdir,
                        str(i)+'.png'
                    )
                )

    x = preprocess(original_paths[idx_start:idx_start+num_images], input_shapes, enforce_detection=False)
    x_attack = attack_fun(model, np.array(x[0]), **params)

    for j in range(num_images):
        a_path = os.path.join(attack_paths[idx_start+j][0], attack_paths[idx_start+j][1])
        os.makedirs(a_path, exist_ok=True)
        a_path = os.path.join(a_path, attack_paths[idx_start+j][2])
        tf.keras.utils.save_img(a_path, x_attack[j])
        attack_paths[idx_start+j] = a_path

    return save_path

def eval_attack(attack_imgs_path):
    # load representations
    train_rep = [
        pickle.load(open('data/lfw-train/representations_vgg_face.pkl', 'rb')),
        pickle.load(open('data/lfw-train/representations_facenet512.pkl','rb'))
    ]

    test_rep = [
        pickle.load(open('data/lfw-test/representations_vgg_face.pkl','rb')),
        pickle.load(open('data/lfw-test/representations_facenet512.pkl','rb'))
    ]

    init_deepface(attack_imgs_path, [model_name, eval_model_name], distance_metric)

    attack_rep = [
        pickle.load(open(attack_imgs_path + '/representations_vgg_face.pkl','rb')),
        pickle.load(open(attack_imgs_path + '/representations_facenet512.pkl','rb'))
    ]

    def path_to_name(p):
        return os.path.split(os.path.split(p)[0])[-1]

    x_train = [np.array([x[1] for x in train_rep[i]]) for i in range(2)]
    y_train = [[path_to_name(x[0]) for x in train_rep[i]] for i in range(2)]
    #----------------------------------------------
    x_test = [np.array([x[1] for x in test_rep[i]]) for i in range(2)]
    y_test =  [[path_to_name(x[0]) for x in test_rep[i]] for i in range(2)]
    #----------------------------------------------
    x_attack = [np.array([x[1] for x in attack_rep[i]]) for i in range(2)]
    y_attack = [[path_to_name(x[0]) for x in attack_rep[i]] for i in range(2)]

    # benchmark: test vs train 
    test_dist = [pairwise_distances(x_test[i], x_train[i]) for i in range(2)]
    # accuracy: attack vs train
    attack_dist = [pairwise_distances(x_attack[i], x_train[i]) for i in range(2)] 

    def classify(ys, ypreds, dist):
        correct = []
        for i, row in enumerate(dist):
            ypred = ypreds[row.argmin()]
            
            correct.append(int(ypred == ys[i]))
        return correct

    test_correct = [classify(y_test[i], y_train[i], test_dist[i]) for i in range(2)]

    attack_correct = [classify(y_attack[i], y_train[i], attack_dist[i]) for i in range(2)] 

    print([(sum(test_correct[i]), sum(test_correct[i])/len(test_correct[i]))for i in [0,1]])
    print([(sum(attack_correct[i]), sum(attack_correct[i])/len(attack_correct[i])) for i in range(2)])
    
def gridsearch():

    # tf automatically uses gpus if found
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print('getting model',model_name)
    model = get_model(model_name)
    eval_model = get_model(eval_model_name)
    
    # run on the test dataset
    test_dataset = os.path.join('data', 'lfw-test')

    input_shapes = (
        DeepFace.functions.find_input_shape(model), DeepFace.functions.find_input_shape(eval_model)
    )
    
    print('initializing deepface on training data')
    init_deepface('data/lfw-train', [model_name, eval_model_name], distance_metric)

    save_path = os.path.join('data', 'lfw-attack-3')
    os.makedirs(save_path, exist_ok=True)

    # run on the test dataset
    test_dataset = os.path.join('data', 'lfw-test')

    ## gridsearch
    attacks = {
        'fgm' : fast_gradient_method,
        'pgd' : projected_gradient_descent,
        # 'cwg' : carlini_wagner_l2
    }

    # grid search
    attack_params = {
        'fgm' : ParameterGrid({
            'eps' : [1e-5,1e-4,1e-3,1e-2],
            'norm' : [1,2,np.inf]
        }),
        'pgd' : ParameterGrid({
            'eps' : [1e-5,1e-4,1e-3,1e-2,1e-1],
            'eps_iter' : [1e-2,1e-1],
            'norm' : [2,np.inf],
            'nb_iter' : [100],
        }),
        # 'cwg' : ParameterGrid({})
    }

    num_options = [len(p) for p in attack_params.values()]
    print('num options:',max(num_options))

    attack_paths = {i : [] for i in attacks}

    for y, subdir in enumerate(os.listdir(test_dataset)):
        person_path = os.path.join(test_dataset, subdir)
        if os.path.isdir(person_path):
            for i, img_file in enumerate(os.listdir(person_path)):
                img_path = os.path.join(person_path, img_file)

                original_paths.append(img_path)    
                
                for a in attack_paths:
                    attack_paths[a].append((
                        save_path,
                        subdir,
                        str(i)+'_'+a+'.png'
                        )
                    )

    print('num batches:', len(original_paths)//batch_size +1)

    # results {
    #   attack : [batch_idx : (model: [name, prob], eval: [name, prob])] 
    # }
    # for a in attacks:
    #     results[a] = []
    # results['original'] = []
    # results['y'] = []
    # do a hyper-param grid search over the batches
    # (instead of over all the data)
    for batch_idx in range(max(num_options)):
        i = batch_idx*batch_size

        # for a in attacks: 
        #     results[a].append(([],[]))
        # results['original'].append(([],[]))
            # results[a].append((0,0))

        tic = time.time()

        x = preprocess(original_paths[i:i+batch_size], input_shapes, enforce_detection=False)
        
        # og_results = (
        #     find(x[0], 
        #         train_rep, 
        #         model=model, model_name=model_name,
        #         distance_metric=distance_metric,
        #         enforce_detection=False
        #     ),
        #     find(x[1], 
        #         eval_train_rep, 
        #         model=eval_model, model_name=eval_model_name,
        #         distance_metric=distance_metric,
        #         enforce_detection=False
        #     ),
        # )
        
        # print(og_results)
        # print(type(og_results), type(og_results[0]),type(og_results[0][0]),type(og_results[0][0][0]))

        x_attacks = {}
        # attack_results = {}

        for a, attack_fun in attacks.items():
            # allow differently sized grid params
            if len(attack_params[a]) <= batch_idx:
                continue
           
            params = attack_params[a][batch_idx]
            param_str = a + '_' + '_'.join([str(params[k]) for k in sorted(list(params.keys()))])
            
            # fix up pgd eps_iter
            if a == 'pgd':
                params['eps_iter'] = params['eps'] * params['eps_iter']
            
            # TODO: checkpoint attack images
            x_attacks[a] = attack_fun(model, np.array(x[0]), **params)

            for j in range(batch_size):
                a_path = os.path.join(attack_paths[a][i+j][0], param_str, attack_paths[a][i+j][1])
                os.makedirs(a_path, exist_ok=True)
                a_path = os.path.join(a_path, attack_paths[a][i+j][2])
                tf.keras.utils.save_img(a_path, x_attacks[a][j])
                attack_paths[a][i+j] = a_path

            #     x_attack = preprocess(x_attacks[a].numpy(), input_shapes, enforce_detection=False)
            # else:
            #     x_attack = preprocess(attack_paths[a][i:i+batch_size], input_shapes, enforce_detection=False)

            # attack_results[a] = (
            #     find(x_attack[0], 
            #         train_rep, 
            #         model=model, model_name=model_name,
            #         distance_metric=distance_metric,
            #         enforce_detection=False
            #     ),
            #     find(x_attack[1], 
            #         eval_train_rep, 
            #         model=eval_model, model_name=eval_model_name,
            #         distance_metric=distance_metric,
            #         enforce_detection=False
            #     ),
            # )

        # vector representations
        # rep_og = model(x)[0]
        # rep_fgm = model(x_fgm)[0]
        # rep_pgd = model(x_pgd)[0]
        # rep_cwg = model(x_cwg)[0]

        # compute results
        # top = prediction
        # for j in range(batch_size):
        #     for k, m in enumerate([model_name, eval_model_name]):
        #         p = original_paths[i+j]
        #         name = path.split(path.split(p)[0])[-1]
        #         results['y'].append(name)
                
        #         og_pred_name = path.split(path.split(og_results[k][j][0])[0])[-1]
        #         og_pred = (og_pred_name, og_results[k][j][1])
        #         # print(results['original'][batch_idx])
        #         # print(results['original'][batch_idx][k])
        #         results['original'][batch_idx][k].append(og_pred)

        #         # results['original'][batch_idx][k] += int(name == og_pred_name)
                
        #         for a in attacks:
        #             if len(attack_params[a]) <= batch_idx:
        #                 continue
        #             a_pred = (
        #                 path.split(path.split(attack_results[a][k][j][0])[0])[-1], 
        #                 attack_results[a][k][j][1]
        #             )
        #             # a_pred = attack_results[a][k][j]['identity'][0]
        #             # a_pred_name = path.split(path.split(attack_results[a][k][j]['identity'][0])[0])[-1]

        #             results[a][batch_idx][k].append(a_pred)
        
        # print(batch_idx, [(a, results[a][batch_idx]) for a in results])

        toc = time.time()
        print(batch_idx, 'time: ', toc - tic)

        # with open(os.path.join(os.path.split(original_paths)[0], 'result.txt'), 'w') as f:
        #     for key in results:
        #         f.writelines([key + ' ' + str(batch_idx) + ' ' + str(i)])
        #         f.writelines(results[key])
    
    # print(results)

    return attack_params


if __name__ == "__main__":
    
    attack_fun = projected_gradient_descent
    attack_params = {'eps':0.001, 'eps_iter':0.0001, 'nb_iter':2000, 'norm':2}
    save_path = attack(attack_fun, attack_params)
    eval_attack(save_path)


    # attack_fun = carlini_wagner_l2
    # attack_params = {}
    # attack(attack_fun, attack_params)
    # eval_attack(save_path)

    # attack_params = gridsearch()
    # with open('attack_params.pkl', 'wb') as f:
    #     pickle.dump(attack_params, f)
