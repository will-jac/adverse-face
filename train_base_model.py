
import sys

from data.datasets import load_data, num_classes
from attacks.base_models.resnet50_torch import train_resnet_classifier, pre_train_resnet_yny, train_resnet_yny

if __name__ == '__main__':

    print("training", sys.argv)

    if 'lfw' in sys.argv:
        data_loader = load_data('lfw', True, 'train', 
            batch_size=16, batch_by_people=False, shuffle=True
        )

    if 'resnet_classifier' in sys.argv:

        train_resnet_classifier(data_loader, num_classes['lfw'], 1e-4, 50, True)

    if 'pretrain_resnet_yny' in sys.argv:


        pre_train_resnet_yny(data_loader, num_classes['lfw'], lr=1e-4, epochs=100, save=True)

    if 'resnet_yny' in sys.argv:

        custom_person = load_data('custom', mode='train', batch_size=10, shuffle=True)

        train_resnet_yny(custom_person, data_loader, num_classes['lfw'], lr=1e-4, epochs=25, boost=4, save=True)
