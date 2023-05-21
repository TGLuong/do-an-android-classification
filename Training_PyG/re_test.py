from collections import Counter
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
import torch.nn.functional as F
from torch_geometric.data import Data


num_top = 100


def add_new_nodes(dataset, new_nodes, new_node_label_encodes):
    X_train = pd.read_csv(f'../JsonFilesToMatrix/output/top_{num_top}_APIs/app_api_{num_top}_APIs_train.csv', header=0, index_col='Unnamed: 0')
    y_train = X_train.pop('Label')
    x_train = X_train.to_numpy()

    number_of_new_nodes = new_nodes.shape[0]
    edge_weight = torch.tensor(new_nodes @ x_train.T)
    self_connections = torch.tensor(new_nodes @ new_nodes.T)
    edge_weight = torch.cat([edge_weight, torch.diagonal(self_connections).reshape(-1, 1)], dim=1)
    max_elements, max_idxs = torch.max(edge_weight, 1)

    edge_weight = torch.div(edge_weight, max_elements.reshape(-1,1) + 0.001)
    edge_weight = edge_weight.flatten()

    new_node_index_start = dataset.edge_index.max().item() + 1
    index_0 = []
    index_1 = []
    for i in range(number_of_new_nodes):
        index_0.extend([new_node_index_start + i] * (new_node_index_start + 1))
        index_1.extend(list(range(0, new_node_index_start)) + [new_node_index_start + i])

    edge_index = torch.tensor([index_0, index_1])

    # add new nodes to graph dataset
    dataset.x = torch.cat([dataset.x, torch.from_numpy(new_nodes).float()], dim=0)
    dataset.edge_index = torch.cat([dataset.edge_index, edge_index], dim=1)
    dataset.weight = torch.cat([dataset.weight, edge_weight], dim=-1)
    dataset.y = torch.cat([dataset.y, new_node_label_encodes], dim=-1)

    return new_node_index_start, dataset


# full_dataset = torch.load(f'./output/full_dataset_{num_top}.pt')

# load x_test
X = pd.read_csv(f'../JsonFilesToMatrix/output/top_{num_top}_APIs/app_api_{num_top}_APIs_test.csv', header=0, index_col='Unnamed: 0')
y_test = X.pop('Label')

with open(f'./output/encoder_{num_top}.pkl', 'rb') as fp:
    encoder = pickle.load(fp)

test_label_encode = torch.from_numpy(encoder.transform(y_test))
x_test = X.to_numpy()

# reload x_train to create edge_weight
X_train = pd.read_csv(f'../JsonFilesToMatrix/output/top_{num_top}_APIs/app_api_{num_top}_APIs_train.csv', header=0, index_col='Unnamed: 0')
y_train = X_train.pop('Label')
x_train = X_train.to_numpy()

# edge_weight
edge_weight = torch.tensor(x_test @ x_train.T)
self_connections = torch.tensor(x_test @ x_test.T)
edge_weight = torch.cat([edge_weight, torch.diagonal(self_connections).reshape(-1, 1)], dim=1)

# standardize edge_weight
max_elements, max_idxs = torch.max(edge_weight, 1)
edge_weight = torch.div(edge_weight, max_elements.reshape(-1,1) + 0.001)
edge_weight = edge_weight.flatten()

# edge_index
new_node_index_start = x_train.shape[0] + 1
index_0 = []
index_1 = []
for i in range(x_test.shape[0]):
    index_0.extend([new_node_index_start + i] * (new_node_index_start + 1))
    index_1.extend(list(range(0, new_node_index_start)) + [new_node_index_start + i])

edge_index = torch.tensor([index_0, index_1])

test_data = Data(x=torch.from_numpy(x_test).float(), y=test_label_encode, edge_index=edge_index, edge_weight=edge_weight)

# Train with x_val
# new_node_index_start, predict_dataset = add_new_nodes(full_dataset, x_test, test_label_encode)


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(-1, 32, aggr='mean', normalize=True)
        self.conv2 = GraphConv(-1, 5, aggr='mean', normalize=True)

    def forward(self, data):
        x, edge_index, weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
model.load_state_dict(torch.load(f"./output/model_{num_top}.pth"))

model.eval()
pred = model(test_data).argmax(dim=1)
correct = (pred[new_node_index_start:] == test_label_encode).sum()
acc = int(correct) / test_label_encode.shape[0]
print(f'Accuracy: {acc:.4f}')


# result = torch.cat([pred[new_node_index_start:].reshape(1, -1), test_label_encode.reshape(1, -1)], dim=0)
# result_np = result.numpy()
# result_df = pd.DataFrame(result_np)
# result_df.to_csv('./output/prediction.csv')
