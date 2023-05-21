from collections import Counter
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
import torch.nn.functional as F


dataset = torch.load('./output/dataset_600.pt')
new_node_index_start = dataset.edge_index.max().item() + 1

X_test = pd.read_csv('./input/app_api_4000.csv', header=0, index_col='Unnamed: 0')
X_test = X_test.iloc[2000:4000]
print(f'Shape: {X_test.shape}')
y_test = X_test.pop('Label')

with open('./output/encoder_600.pkl', 'rb') as fp:
    encoder = pickle.load(fp)
test_label_encode = torch.from_numpy(encoder.transform(y_test))
del y_test

x_test = X_test.to_numpy()
del X_test
number_of_new_nodes = x_test.shape[0]

X_train = pd.read_csv('./input/app_api_600.csv', header=0, index_col='Unnamed: 0')
y_train = X_train.pop('Label')
x_train = X_train.to_numpy()
del X_train

edge_weight = torch.tensor(x_test @ x_train.T)
max_elements, max_idxs = torch.max(edge_weight, 1)
edge_weight = torch.div(edge_weight, max_elements.reshape(-1,1))
edge_weight = edge_weight.flatten()

index_0 = []
index_1 = []
for i in range(number_of_new_nodes):
    index_0.extend([new_node_index_start + i] * new_node_index_start)
    index_1.extend(list(range(0, new_node_index_start)))

edge_index = torch.tensor([index_0, index_1])
del index_0, index_1

#add new nodes to graph dataset
dataset.x = torch.cat([dataset.x, torch.from_numpy(x_test).float()], dim=0)
dataset.edge_index = torch.cat([dataset.edge_index, edge_index], dim=1)
dataset.weight = torch.cat([dataset.weight, edge_weight], dim=-1)
dataset.y = torch.cat([dataset.y, test_label_encode], dim=-1)

del x_test, x_train, edge_index, edge_weight


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


model = GCN()
model.load_state_dict(torch.load("./output/model_600.pth"))

model.eval()
pred = model(dataset).argmax(dim=1)
correct = (pred[new_node_index_start:] == test_label_encode).sum()
acc = int(correct) / test_label_encode.shape[0]
print(f'Accuracy: {acc:.4f}')

result = torch.cat([pred[new_node_index_start:].reshape(1, -1), test_label_encode.reshape(1, -1)], dim=0)
result_np = result.numpy()
result_df = pd.DataFrame(result_np)
result_df.to_csv('./output/2000_4000_prediction.csv')

# 2000: 0.5445
# 2000: 0.7210
# ==> 0.63275