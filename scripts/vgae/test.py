from model import MultiGraphVGAE ,load_model, load_model_from_path
import torch
import args
from torch_geometric.loader import DataLoader
import torch
from model import MultiGraphVGAE
from metrics import calculate_metrics
import logging

logger = logging.getLogger(__name__)

def run_testing(dataset, num_features=93, model_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running test with the following parameters: num_features: {num_features}, model_path: {model_path}, device: {device}")
    
    model = MultiGraphVGAE(num_features).to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if model_path is None:
        load_model(model, model_name=args.model_name, models_dir=args.save_path)
    else:
        load_model_from_path(model, model_path)

    accuarcy, recall, precision, f1, fpr = calculate_metrics(model, loader)
    logger.info(f"Testing Results | Accuracy: {accuarcy}, Recall: {recall}, Precision: {precision}, F1: {f1}, FPR: {fpr}")



def generate_synthetic_graph(model, num_nodes, num_features,loader):
    model.eval()
    with torch.no_grad():
        # Generate random node features
        x = torch.eye(num_nodes)
        
        # Create a dummy edge_index (fully connected graph)
        edge_index = loader.dataset[0].edge_index
        
        # Generate latent representation
        z = model.encode(x, edge_index)
        
        # Decode to get edge probabilities
        edge_attr = model.decoder(z, edge_index)
        
        # Convert probabilities to binary edges (you may want to adjust the threshold)
        edge_mask = edge_attr.squeeze() > 0.5
        synthetic_edge_index = edge_index[:, edge_mask]
        
    return synthetic_edge_index



#synthetic_edge_index = generate_synthetic_graph(model,93,93,loader)
# print(loader.dataset[0].edge_index)
# print(synthetic_edge_index)

