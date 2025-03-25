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
    """Load an image and create a graph representation."""
    image = preprocess_image(image_path)
    graph_data = create_graph_from_image(image)
    return graph_data
