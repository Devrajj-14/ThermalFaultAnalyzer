import cv2
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

def preprocess_image(image_path):
    """Convert image to grayscale and normalize pixel values."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))  # Resize for consistency
    image = image / 255.0  # Normalize pixel values
    return image

def create_graph_from_image(image):
    """Convert an image to a graph representation."""
    height, width = image.shape
    G = nx.grid_2d_graph(height, width)  # Create a grid graph

    # ✅ Convert edges into a 2D tensor correctly
    edge_index = torch.tensor(np.array(G.edges).T, dtype=torch.long)

    # ✅ Ensure node features are formatted correctly
    node_features = image.flatten().reshape(-1, 1)
    x = torch.tensor(node_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index)

def load_graph_data(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    nodes = img.flatten().astype(np.float32) / 255.0
    x = torch.tensor(nodes, dtype=torch.float32).unsqueeze(1)  # shape [4096, 1]

    # Dummy example for edge_index (you should replace this with your graph logic)
    edge_index = []
    for i in range(4096 - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

    # Add node positions for visualization
    pos = []
    for i in range(64):
        for j in range(64):
            pos.append([j, i])
    pos = torch.tensor(pos, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, pos=pos)
