import torch
from torch import nn
import os

class NeuralNet(nn.Module):
    def __init__(self, in_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(in_features=in_size, out_features=468)
        self.fc2 = nn.Linear(468, 256)
        self.fc3 = nn.Linear(256, 150)
        self.fc4 = nn.Linear(150, 68)
        self.fc5 = nn.Linear(68, num_classes)
        self.relu = nn.ReLU()

    def forward(self, X):
        out = self.relu(self.fc1(X))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.relu(self.fc4(out))
        out = self.fc5(out)
        return out

# Define the Keypoint Classification class
class KeypointClassification:
    def __init__(self, path_model):
        self.path_model = path_model
        self.classes = ['Attack', 'Neutral', 'Suspicious']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self):
        self.model = NeuralNet(34, 3).to(self.device)
        self.model.load_state_dict(
            torch.load(self.path_model, map_location=self.device)
        )
        self.model.eval()  # Set model to evaluation mode

    def __call__(self, input_keypoint):
        if len(input_keypoint) != 34:
            print(f"Invalid input keypoint length: {len(input_keypoint)}. Expected 34.")
            return None

        if not isinstance(input_keypoint, torch.Tensor):
            input_keypoint = torch.tensor(input_keypoint, dtype=torch.float32)

        input_keypoint = input_keypoint.to(self.device).unsqueeze(0)

        with torch.no_grad():  # Disable gradient computation for inference
            out = self.model(input_keypoint)
            pred = torch.argmax(out, dim=1).item()
        return self.classes[pred]

      






        
