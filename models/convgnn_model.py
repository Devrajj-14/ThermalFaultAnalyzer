import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class SolarPanelConvGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SolarPanelConvGNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.conv2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
