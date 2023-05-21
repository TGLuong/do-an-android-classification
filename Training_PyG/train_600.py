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
    adj_matr = adj_matr / np.amax(adj_matr, axis=1)
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


X = pd.read_csv('../JsonFilesToMatrix/output/app_api_3000_50_APIs.csv', header=0, index_col='Unnamed: 0')
y = X.pop('Label')
label_encoder = preprocessing.LabelEncoder().fit(y)

x = X.to_numpy()

edge_index, weight = create_edge('./input/m1_600.csv')
dataset = Data(x=torch.from_numpy(x).float(), y=torch.from_numpy(label_encoder.transform(y)), edge_index=edge_index, weight=weight)

train_dataset, val_dataset, test_dataset = random_split(x, [1000, 500, 1500], torch.Generator().manual_seed(42))

train_mask = [i in train_dataset.indices for i in range(len(x))]
val_mask = [i in val_dataset.indices for i in range(len(x))]
test_mask = [i in test_dataset.indices for i in range(len(x))]

dataset.train_mask = torch.asarray(train_mask)
dataset.val_mask = torch.asarray(val_mask)
dataset.test_mask = torch.asarray(test_mask)

# dataset.train_mask = torch.cat([dataset.train_mask, torch.tensor([False]*number_of_new_nodes)], dim=-1)
# dataset.val_mask = torch.cat([dataset.val_mask, torch.tensor([False]*number_of_new_nodes)], dim=-1)
# dataset.test_mask = torch.cat([dataset.test_mask, torch.tensor([False]*number_of_new_nodes)], dim=-1)

# print(Counter(dataset.y[dataset.train_mask].tolist()))
# print(Counter(dataset.y[dataset.val_mask].tolist()))
# print(Counter(dataset.y[dataset.test_mask].tolist()))

torch.save(dataset, './output/dataset_600.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

data = dataset
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    acc = int(correct) / int(data.val_mask.sum())
    print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')


# Saving
torch.save(model.state_dict(), './output/model_600.pth')

with open('./output/encoder_600.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Epoch: 99, Accuracy: 0.9540
# Accuracy: 0.9587

# Epoch: 99, Accuracy: 0.9640
# Accuracy: 0.9707