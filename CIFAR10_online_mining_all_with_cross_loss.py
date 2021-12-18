# %%
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

from TNN import Mining, Model
from TNN.Plot import scatter
from TNN.Loss_Fn import triplet_loss
from torch import nn
# %%
device = torch.device('cuda:0')
tsne = TSNE(random_state=0)
batch_size_train = 64
batch_size_vis = 256

# %%
train_loader = DataLoader(CIFAR10('./CIFAR10/', train=True, download=True,
                                transform=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                               ),batch_size=batch_size_vis, shuffle=True)
test_loader = DataLoader(CIFAR10('./CIFAR10/', train=False, download=True,
                                transform=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                               ),batch_size=batch_size_vis, shuffle=False)

# %%
x_test, y_test = next(iter(test_loader))
x_test = x_test.to(device)
y_test = y_test.to(device)

# %%
x_train, y_train = next(iter(train_loader))
x_train = x_train.to(device)
y_train = y_train.to(device)

# %%
train_loader = DataLoader(CIFAR10('./CIFAR10/', train=True, download=True,
                                transform=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                               ),batch_size=batch_size_train, shuffle=True)
test_loader = DataLoader(CIFAR10('./CIFAR10/', train=False, download=True,
                                transform=Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                               ),batch_size=batch_size_train, shuffle=False)

# %%
train_tsne_embeds = tsne.fit_transform(x_train.flatten(1).cpu().detach().numpy())
test_tsne_embeds = tsne.fit_transform(x_test.flatten(1).cpu().detach().numpy())

# %%
scatter(train_tsne_embeds, y_train.cpu().numpy(), subtitle=f'online Original MNIST distribution (train set)', dataset='CIFAR10')
scatter(test_tsne_embeds, y_test.cpu().numpy(), subtitle=f'online Original MNIST distribution (test set)', dataset='CIFAR10')

# %%
model = Model.TNN_CIFAR10_Drop(input_shape=x_train.shape[1:],output_size=10)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

# %%
model
criterion = nn.CrossEntropyLoss()

# %%
margin = 0.1
for epoch in range(300):  # loop over the dataset multiple times

    running_loss = 0.0
    model.train()
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        # loss, pos_triplet, valid_triplet = Mining.online_mine_all(labels, outputs, margin=margin, squared=True, device=device)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i%2 == 0:
            print(f"At epoches = {epoch}, i = {i}, cross_loss = {loss:.5f}"
                  , end='\r')
    train_epoch_loss = running_loss / len(train_loader)
    # Evaluating on test set
    print(" "*100)
    running_loss = 0.0
    model.eval()

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    for i, data in enumerate(test_loader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # loss, pos_triplet, valid_triplet = Mining.online_mine_all(labels, outputs, margin=margin, squared=True, device=device)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, pred_label = torch.max(outputs.data, 1)

        total += inputs.data.size()[0]
        correct += (pred_label == labels.data).sum().item()

        if device == "cpu":
            pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
            true_labels_list = np.append(true_labels_list, labels.data.numpy())
        else:
            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, labels.data.cpu().numpy())

        # if i%2 == 0:
            # print(f"Evaluating At epoches = {epoch}, i = {i}, loss = {loss:.5f}", end='\r')
    test_epoch_loss = running_loss / len(test_loader)

    acc = correct / float(total)

    print(f"At epoches = {epoch}, epoch_train_loss = {train_epoch_loss:.6f},\tepoch_test_loss = {test_epoch_loss:.6f},\tepoch_test_acc= {acc:.6f}")
print('Finished Training')

# %%
train_outputs = model(x_train)
test_outputs = model(x_test)
train_tsne_embeds = tsne.fit_transform(train_outputs.cpu().detach().numpy())
test_tsne_embeds = tsne.fit_transform(test_outputs.cpu().detach().numpy())

scatter(train_tsne_embeds, y_train.cpu().numpy(), subtitle=f'online TNN distribution (train set)', dataset='CIFAR10')
scatter(test_tsne_embeds, y_test.cpu().numpy(), subtitle=f'online TNN distribution (test set)', dataset='CIFAR10')

# %%


# %%
