import torch.nn as nn
import torch.nn.functional as F
import torch
import os 

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim,model_name="Model"):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) 
        self.model_name = model_name

    def forward(self, embedding1, embedding2):
        # Flatten the embeddings
        embedding1 = embedding1.view(embedding1.size(0), -1)
        embedding2 = embedding2.view(embedding2.size(0), -1)
        # Calculate absolute difference
        combined = torch.abs(embedding1 - embedding2)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output probability
        return x

        
def save_model(model):
  """Saves a PyTorch model to the specified path.

  Args:
      model: The PyTorch model to save.
  """

  # Create the models directory if it doesn't exist
  models_dir = os.path.join('./', 'models')  # Adjust path if needed
  if(not os.path.exists(models_dir)):
      os.mkdir(models_dir)
      
  os.makedirs(models_dir, exist_ok=True)  # Create directory if it doesn't exist

  # Construct the filename with model name and .pt extension
  filename = f"{model.model_name}.pt"

  # Create the full path to the model file
  path = os.path.join(models_dir, filename)

  # Save the model state dictionary
  torch.save(model.cpu().state_dict(), path)
  print(f'Model saved to {path}')