import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = nn.Dropout(0.5)(self.pool(F.relu(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(n_train_samples: int = 1000, n_val_samples: int = 500) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_train_dataset = MNIST('./', train=True, download=False, transform=transform)
    batch_size = 64
    indices = np.arange(20000)
    np.random.shuffle(indices)
    train_dataset = Subset(full_train_dataset, torch.from_numpy(indices[:n_train_samples]))
    val_dataset = Subset(full_train_dataset, torch.from_numpy(indices[n_train_samples:n_train_samples+n_train_samples]))

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True)
    return train_dl, val_dl


def load_test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = MNIST('./', train=False, download=False, transform=transform)
    return test_dataset


def train(model: Net, num_epochs: int, train_dl: DataLoader) -> Tuple[list, list]:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_hist_train = [0] * num_epochs
    acc_hist_train = [0] * num_epochs
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        counter = 0
        tk0 = tqdm(train_dl, total=int(len(train_dl)))
        for x_batch, y_batch in tk0:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_hist_train[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()
            acc_hist_train[epoch] += is_correct.sum()

            counter += 1
            tk0.set_postfix(loss=(loss/(counter*y_batch.size(0))).item(),
                            accuracy=(acc_hist_train[epoch]/(counter*train_dl.batch_size)).item())
        loss_hist_train[epoch] /= len(train_dl.dataset)
        acc_hist_train[epoch] /= len(train_dl.dataset)
        tk0.set_postfix(loss=loss.item(),
                            accuracy=acc_hist_train[epoch].item())
    return loss_hist_train, acc_hist_train



def test(model: Net, test_dataset: DataLoader) -> float:
    data = test_dataset.data
    targets = test_dataset.targets
    y_pred = model(data.unsqueeze(1)/255.)
    is_correct = (torch.argmax(y_pred, dim=1)==targets).float()
    acc = is_correct.mean()
    return float(acc.detach())


def evaluate(model: Net, val_dl: DataLoader) -> Tuple[float, float]:
    loss_fn = nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            acc = ((torch.argmax(y_pred, dim=1) == y_batch).float()).mean()
    return float(loss), float(acc)

if __name__ == '__main__':
    net = Net()
    torch.save(net.state_dict(), 'global_parameters.pth')
    # train_dl, val_dl = load_data()
    # # train(net, 5, train_dl)
    # res = evaluate(net, val_dl)
    # print(res)
