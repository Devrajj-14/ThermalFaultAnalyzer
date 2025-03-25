import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from utils.preprocess import load_graph_data
from models.convgnn_model import SolarPanelConvGNN

image_folder = "./data/thermal_images/"

# ✅ Ensure the folder exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Error: The folder '{image_folder}' does not exist. Please check your dataset path.")

# ✅ Ensure images exist
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpeg")]

if not image_files:
    raise FileNotFoundError(f"Error: No images found in '{image_folder}'. Please add dataset images.")

# ✅ Convert images to graphs
graph_data_list = [load_graph_data(img_path) for img_path in image_files]

# ✅ Ensure graphs were created
if not graph_data_list:
    raise ValueError("Error: No graph data was created. Check if images are correctly processed.")

# ✅ Create DataLoader
train_loader = DataLoader(graph_data_list, batch_size=4, shuffle=True)

# ✅ Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarPanelConvGNN(in_channels=1, hidden_channels=16, out_channels=2).to(device)

# ✅ Loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ✅ Training function
def train(model, loader, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index)
            labels = torch.randint(0, 2, (data.x.size(0),)).to(device)  # Random labels for demo
            loss = F.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(loader)}")

# ✅ Train the model
train(model, train_loader)

# ✅ Save the trained model
torch.save(model.state_dict(), "models/solar_panel_gnn.pth")
print("Model training complete. Model saved.")