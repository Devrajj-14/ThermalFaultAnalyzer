import torch
import torch.nn.functional as F
from utils.preprocess import load_graph_data
from models.convgnn_model import SolarPanelConvGNN
from collections import Counter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SolarPanelConvGNN(in_channels=1, hidden_channels=16, out_channels=4).to(device)


model_path = "models/solar_panel_gnn.pth"
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f" Error loading model: {e}")
    exit()


test_image_path = "data/Unknown.png"

try:
    graph_data = load_graph_data(test_image_path).to(device)
    print(f"Test image '{test_image_path}' successfully processed into graph.")
except Exception as e:
    print(f"Error processing image: {e}")
    exit()


with torch.no_grad():
    output = model(graph_data.x, graph_data.edge_index)
    probabilities = F.softmax(output, dim=1)
    predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
    majority_vote = Counter(predicted_labels).most_common(1)[0][0]
    confidence_score = probabilities[:, majority_vote].mean().item() * 100


fault_types = {
    0: "No Fault (Normal)",
    1: "Panel Fault (Defective)",
    2: "Hotspot Detected",
    3: "Cracked Panel"
}
predicted_fault = fault_types.get(majority_vote, "Unknown Fault Type")


print("\n **Prediction Results**:")
print(f" Test Image: {test_image_path}")
print(f" Majority Predicted Label: {majority_vote}")
print(f" Fault Type: {predicted_fault}")
print(f" Confidence Score: {confidence_score:.2f}%")
