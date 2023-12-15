import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import torch.optim as optim

num_timesteps_input = 30
num_nodes = 5
num_features_per_node = 5  # Open, High, Low, Close, Volume

class STGCN(nn.Module):
    def __init__(self, num_nodes, num_features_per_node, num_timesteps_input):
        super(STGCN, self).__init__()
        self.gcn = GCNConv(num_features_per_node * num_timesteps_input, 16)
        self.tcn = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 3))
        self.fc = nn.Linear(num_nodes * (16-3+1), num_nodes)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn(x, edge_index))
        x = x.view(-1, num_nodes, 16).unsqueeze(1)
        x = F.relu(self.tcn(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

stock_data = pd.read_csv('stock.csv', index_col='Date')
weight_matrix = pd.read_csv('weight.csv', header=None).to_numpy()

scaler = MinMaxScaler()
stock_data_normalized = scaler.fit_transform(stock_data)

temporal_features = []
targets = []
for i in range(len(stock_data_normalized) - num_timesteps_input - 1):
    targets.append(stock_data_normalized[i + num_timesteps_input, 3::5])
    window = stock_data_normalized[i:i + num_timesteps_input]
    window_features = window.reshape(num_timesteps_input, num_nodes, -1).transpose(1, 0, 2).reshape(num_nodes, -1)
    temporal_features.append(window_features)

targets = np.array(targets)
targets = torch.tensor(targets, dtype=torch.float)

temporal_features = np.array(temporal_features)
temporal_features = torch.stack([torch.tensor(features, dtype=torch.float) for features in temporal_features])

max_value = np.max(weight_matrix[np.nonzero(weight_matrix - np.eye(num_nodes))])
normalized_weight_matrix = weight_matrix / max_value
np.fill_diagonal(normalized_weight_matrix, 1)
edge_index = torch.tensor([[], []], dtype=torch.long)
weights = torch.tensor(normalized_weight_matrix.flatten(), dtype=torch.float)

for i in range(num_nodes):
    for j in range(num_nodes):
        edge_index = torch.cat([edge_index, torch.tensor([[i], [j]], dtype=torch.long)], dim=1)

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15
X_train, X_temp, y_train, y_temp = train_test_split(temporal_features, targets, test_size=1 - train_ratio)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_ratio / (test_ratio + validation_ratio))

data_train = Data(x=X_train, edge_index=edge_index, edge_attr=weights)
data_val = Data(x=X_val, edge_index=edge_index, edge_attr=weights)
data_test = Data(x=X_test, edge_index=edge_index, edge_attr=weights)

model = STGCN(num_nodes=num_nodes, num_features_per_node=num_features_per_node, num_timesteps_input=num_timesteps_input)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data_train.x, data_train.edge_index)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    output = model(data_test.x, data_test.edge_index)
    loss = criterion(output, y_test)
    print(f'test loss: {loss}')



# model.load_state_dict(torch.load('model.pth'))
# torch.save(model.state_dict(), 'model.pth')
# def predict(model, data):
#     model.eval()
#     with torch.no_grad():
#         output = model(data.x, data.edge_index)
#     return output
# data_predict = Data(x=X_predict, edge_index=edge_index, edge_attr=weights)
# predictions = predict(model, data_predict)


