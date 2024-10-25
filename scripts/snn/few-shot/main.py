import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SiameseNetwork, save_model
from dataset import SiameseDataset
from test import test_siamese_network
import args
from train import train_siamese_network

def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    # Load the saved embeddings
    embeddings_path = args.embeddings_path
    embeddings = torch.load(embeddings_path, map_location=torch.device('cpu'))

    # Create the Siamese dataset
    siamese_dataset = SiameseDataset(embeddings)
    siamese_loader = DataLoader(siamese_dataset, batch_size=args.batch, shuffle=True)

    # Initialize the SNN model
    embedding_dim = embeddings[0][0].view(-1).size(0)  # Adjusted to the flattened dimension
    snn_model = SiameseNetwork(embedding_dim).to(device)

    # Example usage
    optimizer = torch.optim.Adam(snn_model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()

    # Train the Siamese Network
    train_siamese_network(snn_model, siamese_loader, optimizer, criterion, device)
    # Example usage
    test_embeddings_path = args.test_embeddings_path
    support_set_path = args.support_set_path

    test_siamese_network(snn_model, test_embeddings_path, support_set_path ,criterion, device)
    save_model(snn_model)

if __name__ == "__main__":
    main()