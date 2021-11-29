
from attacks.obfuscated_gradient.baseline_torch import attack

import torch
import numpy as np

from matplotlib.pylab import plt

if __name__ == '__main__':

    # load data
    from data.datasets import load_data, num_classes
    data_loader = load_data('lfw', True, 'train', 
        batch_size=10, batch_by_people=True, shuffle=False
    )

    for image_batch, label_batch in data_loader:
        # print(image_batch.shape)
        image = image_batch[0]
        label = label_batch[0]
        break

    image = image.reshape([1] + list(image.shape))
    print("label:", label)
    # print(image.shape)
    # image = image.permute(1,2,0)
    # print(image.shape)

    # do cifar10 or vgg16 as base model
    from attacks.base_models.resnet50_torch import load_classifier
    model = load_classifier()

    logits = model.forward(image)
    print("logits:", logits.shape)

    # import torch.nn.functional as F
    # print("softmax:", F.softmax(logits, dim=1).shape)
    # print("sigmoid:", F.softmax(logits).shape)
    # print("predicted label:", F.softmax(logits))
    
    # from torchsummary import summary
    # summary(model)

    # attack = Attack(model,1,100,1,False)

    adversarial = attack(image, label, model, num_classes=num_classes['lfw'], num_steps=1000)

    import torchvision
    torchvision.utils.save_image(adversarial, "test.png")

    # from PIL import Image

    # result = Image.fromarray((adversarial * 255).astype(np.uint8))
    # result.save('test.png')

    # image = cifar.eval_data.xs[:1]
    # label = cifar.eval_data.ys[:1]

    # plt.imshow(image[0]/255.0)
    # plt.show()
    print("Image Label", label)

    logits = model.forward(image)

    print('Clean Model Prediction', logits)
    # print('\tLogits', pre_softmax)
    print('classification:', logits.argmax())

    # abs_distortion = np.abs(adversarial/255.0-image/255.0)
    # print(abs_distortion)
    print("Max distortion", (adversarial-image).abs().max())

    logits = model.forward(adversarial)

    print('Adversarial Model Prediction', logits)
    # print('\tLogits', pre_softmax)
    print('classification:', logits.argmax())
