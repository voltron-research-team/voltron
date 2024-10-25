import torch


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


