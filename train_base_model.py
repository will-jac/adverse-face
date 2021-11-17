
import sys

if __name__ == '__main__':

    print("training", sys.argv)

    if 'lfw' in sys.argv:
        from data.datasets import load_data, num_classes
        data_loader = load_data('lfw', True, 'train', 
            batch_size=16, batch_by_people=False, shuffle=True
        )

    if 'resnet_classifier' in sys.argv:

        from attacks.base_models.resnet50_torch import train_resnet_classififier

        train_resnet_classififier(data_loader, num_classes['lfw'], 1e-4, 50, True)

    if 'resnet_yny' in sys.argv:

        from attacks.base_models.resnet50_torch import train_resnet_yny

        you_data_loader = load_data('lfw', True, 'test', 
            batch_size=16, batch_by_people=True, shuffle=False
        )
        
        

        train_resnet_yny(data_loader, num_classes['lfw'], 1e-4, 50, True)
