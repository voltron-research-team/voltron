from torch_geometric.nn import global_mean_pool
import torch
from torch_geometric.nn import GCNConv, VGAE
from torch.nn import functional as F
from torch.nn import Linear
import time
import os
import logging

class MultiGraphVGAE(VGAE):
    def __init__(self, num_features, model_name="Model"):
        hidden_channels1 = 32
        hidden_channels2 = 24
        encoder = GCNEncoder(num_features, hidden_channels1, hidden_channels2, 16)
        super(MultiGraphVGAE, self).__init__(encoder)
        self.classifier = Linear(16, 2)
        self.model_name = model_name

    def classify(self, z, batch):
        # Global mean pooling
        z = global_mean_pool(z, batch)
        return self.classifier(z)

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels1)
        self.conv2 = GCNConv(hidden_channels1, hidden_channels2)
        self.conv_mu = GCNConv(hidden_channels2, out_channels)
        self.conv_logstd = GCNConv(hidden_channels2, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # Added new layer with ReLU activation
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


def check_save_conditions(model, models_dir):
    """Checks the conditions necessary for saving a PyTorch model and creates the directory if it doesn't exist.

        Args:
            model: The PyTorch model that will be saved.
            models_dir: The directory where the model will be saved.
        
        Raises:
            ValueError: If models_dir is not a valid directory.
            PermissionError: If models_dir is not writable.
            AttributeError: If the model does not have a 'model_name' attribute.
            TypeError: If the model is not a valid PyTorch model.
    """
    
    # Check if models_dir is a valid directory
    if not os.path.isdir(models_dir):
        raise ValueError(f"The path {models_dir} is not a valid directory.")

    # Create the directory if it doesn't exist and log info
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logging.info(f"Directory {models_dir} was created.")
    else:
        logging.info(f"Directory {models_dir} already exists.")
    
    # Ensure the directory is writable
    if not os.access(models_dir, os.W_OK):
        raise PermissionError(f"No write permission for directory {models_dir}.")

    # Check if the model has a 'model_name' attribute
    if not hasattr(model, 'model_name'):
        raise AttributeError("The model does not have a 'model_name' attribute.")

    # Check if the model is a valid PyTorch model
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The provided model is not a valid PyTorch model.")
    
    absolute_path = os.path.abspath(models_dir)
    logging.info(f"All save conditions are met: {model.model_name} will be saved to {models_dir} ({absolute_path}).")

    return True


def save_model(model, models_dir, epoch=None):
    """Saves a PyTorch model to the specified path.

        Args:
            model: The PyTorch model to save.
    """
    try:
        # assert if the directory is not a directory
        assert os.path.isdir(models_dir), 'The directory is not a directory'

        os.makedirs(models_dir, exist_ok=True)  # Create directory if it doesn't exist

        model_name = model.model_name

        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pt"
        else:
            filename = f"{model_name}.pt"

        # Create the full path to the model file
        path = os.path.join(models_dir, filename)

        # if path exist, warn the user
        if os.path.exists(path):
            logging.warning(f"Model file {path} already exists and will be overwritten.")

        # Save the model state dictionary
        torch.save(model.cpu().state_dict(), path)
        
        if epoch is not None:
            logging.info(f'Model saved to {path} with name {filename} at epoch {epoch}')
        else:
            logging.info(f'Model saved to {path} with name {filename}')
    except Exception as e:
        logging.error(f"An exception occurred when saving model: {e}", exc_info=True)
        raise e
  
  
def load_model(model, model_name, models_dir):
    """Loads a PyTorch model from the specified path.

    Args:
        model: The PyTorch model instance to load the state dictionary into.
        model_name: The name of the model to load (without .pt extension).
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        filename = f"{model_name}.pt"
        path = os.path.join(models_dir, filename)

        # Load the model state dictionary
        model.load_state_dict(torch.load(path))
        model.model_name = model_name
        model.to(device)
        model.eval()  # Set the model to evaluation mode

        logging.info(f'Model loaded from {path} with name {model_name}')
    except Exception as e:
        logging.error(f"An exception occurred when loading model: {e}", exc_info=True)
        raise e

def load_model_from_path(model, path):
    """Loads a PyTorch model from the specified path.

    Args:
        model: The PyTorch model instance to load the state dictionary into.
    """

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the model state dictionary
        model.load_state_dict(torch.load(path))
        model.to(device)
        model.eval()  # Set the model to evaluation mode

        logging.info(f'Model loaded from {path}')
    except Exception as e:
        logging.error(f"An exception occurred when loading model: {e}", exc_info=True)
        raise e


