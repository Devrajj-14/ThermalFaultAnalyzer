import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utils.preprocess import load_graph_data
from models.convgnn_model import SolarPanelConvGNN

# Paths
IMAGE_FOLDER = "data/thermal_images"
LABELS_PATH = "data/labels.json"
MODEL_PATH = "models/solar_panel_gnn.pth"

# ----------------------------
# 1) LOAD LABELS
# ----------------------------
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"❌ No '{LABELS_PATH}' found. Please create a real labels file.")

with open(LABELS_PATH, "r") as f:
    label_dict = json.load(f)  # { "C24.jpeg": 3, "N10.jpeg": 0, ... }

# ----------------------------
# 2) CREATE DATASET
# ----------------------------
if not os.path.exists(IMAGE_FOLDER):
    raise FileNotFoundError(f"❌ The folder '{IMAGE_FOLDER}' does not exist.")

image_files = [
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not image_files:
    raise FileNotFoundError(f"❌ No images found in '{IMAGE_FOLDER}'.")

graph_data_list = []
for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    # If label not found in your label dict, skip
    if img_file not in label_dict:
        print(f"⚠️  Skipping '{img_file}' (no label in labels.json).")
        continue

    # Convert to graph
    data = load_graph_data(img_path)
    # Repeated node-level labels: each node in this image gets the same label
    class_label = label_dict[img_file]
    data.y = torch.full((data.x.size(0),), class_label, dtype=torch.long)
    graph_data_list.append(data)

if not graph_data_list:
    raise ValueError("❌ No valid graph data created (check labeling).")

train_loader = DataLoader(graph_data_list, batch_size=4, shuffle=True)

# ----------------------------
# 3) INIT MODEL
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarPanelConvGNN(in_channels=1, hidden_channels=16, out_channels=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# 4) TRAIN FUNCTION
# ----------------------------
def train(model, loader, epochs=50):
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            # data.y is shape [num_nodes], same shape as output
            loss = F.cross_entropy(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch:02d}, Loss: {total_loss / len(loader):.4f}")

# ----------------------------
# 5) TRAIN & SAVE
# ----------------------------
train(model, train_loader, epochs=50)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model training complete. Saved at '{MODEL_PATH}'.")
