
import torch
from torch import nn

import numpy as np



if __name__ == '__main__':

    print("training")

    from data.datasets import load_data
    data_loader = load_data('lfw', True, 'train', 
        batch_size=64, batch_by_people=False, shuffle=True
    )

    from attacks.base_models.resnet50_torch import ResNetClassifier

    model = ResNetClassifier()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    for i in range(1000):
        l = 0
        for image_batch, label_batch in data_loader:
            optimizer.zero_grad()

            logits = model(image_batch) 
            # print(logits.shape)
            
            label = nn.functional.one_hot(label_batch, 5749).float()
            # print(label.shape)

            preds = torch.sigmoid(logits)
            loss = loss_fn(preds, label)
            loss.backward()
            optimizer.step()

            l += loss.data.item()
        print('loss:',loss.data.item())