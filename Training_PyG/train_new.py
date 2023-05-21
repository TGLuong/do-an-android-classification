from os import path
from collections import Counter
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.data import Data

from sklearn import preprocessing


num_top = 400


def create_edge_index(node_number):
    #TODO: Loai duong cheo chinh
    index_0 = []
    index_1 = []
    
    for i in range(node_number):
        index_0.extend([i]*(i+1))
        index_1.extend(list(range(0, i+1)))
    
    return torch.tensor([index_0, index_1])


def create_edge(meta_path):
    adj_matr = pd.read_csv(meta_path, index_col=None, header=None)
    node_number = adj_matr.shape[0]
    
    #edge_index
    edge_index = create_edge_index(node_number)
    
    #weight
    adj_matr = adj_matr.to_numpy()
    adj_matr = adj_matr / (np.amax(adj_matr, axis=1) + 0.001)
    adj_matr = np.tril(adj_matr)

    #TODO: Loai duong cheo chinh
    weight = []
    for i in range(node_number):
        weight.extend(adj_matr[i, :i+1])
    
    return edge_index, torch.tensor(weight, dtype=torch.float)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(-1, 32, aggr='mean', normalize=True)
        self.conv2 = GraphConv(-1, 5, aggr='mean', normalize=True)

    def forward(self, data):
        x, edge_index, weight = data.x, data.edge_index, data.weight
        x = self.conv1(x, edge_index, weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if path.isfile(f'./output/dataset_{num_top}.pt'):
    dataset = torch.load(f'./output/full_dataset_{num_top}.pt')
    train_dataset = torch.load(f'./output/train_dataset_{num_top}.pt')
else:
    X = pd.read_csv(f'../JsonFilesToMatrix/output/top_{num_top}_APIs/app_api_{num_top}_APIs_train.csv', header=0, index_col='Unnamed: 0')
    y = X.pop('Label')
    label_encoder = preprocessing.LabelEncoder().fit(y)
    with open(f'./output/encoder_{num_top}.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    x = torch.from_numpy(X.to_numpy()).float()

    edge_index, weight = create_edge(f'../JsonFilesToMatrix/output/top_{num_top}_APIs/m1_{num_top}_APIs.csv')
    dataset = Data(x=x, y=torch.from_numpy(label_encoder.transform(y)), edge_index=edge_index, weight=weight)

    train_dataset, test_dataset = random_split(x, [2400, 600], torch.Generator().manual_seed(42)) # type: ignore
    train_mask = [i in train_dataset.indices for i in range(len(x))]
    test_mask = [i in test_dataset.indices for i in range(len(x))]
    dataset.test_mask = torch.asarray(test_mask)

    train_dataset = dataset.subgraph(torch.asarray(train_mask))
    
    sub_train_dataset, sub_val_dataset = random_split(train_dataset.x, [480, 1920], torch.Generator().manual_seed(42)) # type: ignore
    sub_train_mask = [i in sub_train_dataset.indices for i in range(len(train_dataset.x))]
    sub_val_mask = [i in sub_val_dataset.indices for i in range(len(train_dataset.x))]
    train_dataset.train_mask = torch.asarray(sub_train_mask)
    train_dataset.val_mask = torch.asarray(sub_val_mask)

    torch.save(dataset, f'./output/full_dataset_{num_top}.pt')
    torch.save(train_dataset, f'./output/train_dataset_{num_top}.pt')

epochs = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
test_dataset = dataset
dataset = train_dataset

if path.isfile(f'./output/model_{num_top}.pth'):
    model.load_state_dict(torch.load(f"./output/model_{num_top}.pth"))

model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(dataset)
    loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(dataset).argmax(dim=1)
    correct = (pred[dataset.val_mask] == dataset.y[dataset.val_mask]).sum()
    acc = int(correct) / int(dataset.val_mask.sum())
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

model.eval()
pred = model(test_dataset).argmax(dim=1)
correct = (pred[test_dataset.test_mask] == test_dataset.y[test_dataset.test_mask]).sum()
acc = int(correct) / int(test_dataset.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# Saving
torch.save(model.state_dict(), f'./output/model_{num_top}.pth')
