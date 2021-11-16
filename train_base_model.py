
import torch
from torch import nn

import numpy as np



if __name__ == '__main__':

    print("training")

    torch.cuda.empty_cache()

    from data.datasets import load_data
    data_loader = load_data('lfw', True, 'train', 
        batch_size=16, batch_by_people=False, shuffle=True
    )

    from attacks.base_models.resnet50_torch import load_classifier

    model = load_classifier()

    
    device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    # if device == "cuda":
    model = model.to(device=device)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()

    for i in range(100):
        l = 0
        for x, y in data_loader:
            # print(label_batch, end=' ') 
            y = nn.functional.one_hot(y, 5749).float()
            # print(label.shape)
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            logits = model(x) 
            # print(logits.shape)
            
            preds = logits
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            # print(loss.item())
            l += loss.item()

        print('epoch', i, 'loss:',l)
    
    torch.save(model.state_dict(), './attacks/base_models/ResNet50Classifer.pth')


