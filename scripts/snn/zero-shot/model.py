import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import datetime

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, model_name=None):
        super(SiameseNetwork, self).__init__()
        layers = []
        input_dim = embedding_dim

        # Create layers based on hidden_dims
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, 1))

        self.layers = nn.ModuleList(layers)
        if model_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.model_name = f"siamese_network_{timestamp}"
        else:
            self.model_name = model_name

    def forward(self, embedding1, embedding2):
        # Flatten the embeddings
        embedding1 = embedding1.view(embedding1.size(0), -1)
        embedding2 = embedding2.view(embedding2.size(0), -1)

        # Calculate absolute difference
        combined = torch.abs(embedding1 - embedding2)

        # Pass through the layers
        x = combined
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Final output layer with sigmoid activation
        x = torch.sigmoid(self.layers[-1](x))
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