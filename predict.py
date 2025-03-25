import torch
import torch.nn.functional as F
from utils.preprocess import load_graph_data
from models.convgnn_model import SolarPanelConvGNN

# ✅ Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarPanelConvGNN(in_channels=1, hidden_channels=16, out_channels=2).to(device)

# ✅ Load model weights
try:
    model.load_state_dict(torch.load("models/solar_panel_gnn.pth", map_location=device))
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("❌ Error: Model file 'models/solar_panel_gnn.pth' not found.")
    exit()

# ✅ Load new test image (Replace with actual image)
test_image_path = "data/thermal_images/test_panel.jpg"

try:
    graph_data = load_graph_data(test_image_path).to(device)
    print(f"✅ Test image '{test_image_path}' successfully processed into graph.")
except FileNotFoundError:
    print(f"❌ Error: Test image '{test_image_path}' not found.")
    exit()

# ✅ Perform Prediction
with torch.no_grad():
    output = model(graph_data.x, graph_data.edge_index)
    probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
    predicted_label = torch.argmax(probabilities, dim=1).cpu().numpy()

# ✅ Print Results
print("\n🔍 **Prediction Results**:")
print(f"📌 Test Image: {test_image_path}")
print(f"🔹 Raw Model Output: {output.cpu().numpy()}")
print(f"🔹 Predicted Label: {predicted_label}")
print(f"🔹 Probabilities: {probabilities.cpu().numpy()}")

# ✅ Map Labels to Fault Types (Customize as needed)
fault_types = {0: "No Fault (Normal)", 1: "Panel Fault (Defective)"}
predicted_fault = fault_types.get(predicted_label[0], "Unknown")

print(f"🚀 Final Prediction: {predicted_fault}")
