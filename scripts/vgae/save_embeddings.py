from model import MultiGraphVGAE, load_model, load_model_from_path
import torch
import args
from torch_geometric.loader import DataLoader
from dataset import MultiGraphDataset
import logging
import os
import argparse
from logging_setup import setup_logging
import pandas as pd
from pathlib import Path
from datetime import datetime


# fold == train/test/support_fold

def save_embeddings(path, model_path, fold, num_features=658, embedding_dir='embeddings'):
    # Load multiple graphs
    dataset = MultiGraphDataset(root=path,fold=fold)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a new instance of the model
    model = MultiGraphVGAE(num_features).to(device)

    # Load the model state dictionary into the new instance
    load_model_from_path(model, path=model_path)

    embeddings = list()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for idx, data in enumerate(dataset):  # Get index and data
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            adj_file_path = dataset.adj_files[idx]  # Get corresponding adjacency matrix file path
            embeddings.append((z.cpu(), data.y, adj_file_path))  # Move tensor to CPU before appending

    model_name = Path(model_path).stem
    
    embedding_dir_path = Path(embedding_dir)
    embedding_dir_path.mkdir(parents=True, exist_ok=True)
    
    embedding_path = embedding_dir_path / f'embed_{model_name}_{fold.iloc[0]["fold_label"]}_f{fold.iloc[0]["fold"]}.pth'
    
    if embedding_path.exists():
        # Append current date and time to make it unique
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        embedding_path = embedding_dir_path / f'embed_{model_name}_{fold.iloc[0]["fold_label"]}_f{fold.iloc[0]["fold"]}_{current_time}.pth'
        
        logging.warning(f"File already exists at {embedding_path}. Adding date and time to create a unique path.")
    
    # Save the list of tensors
    torch.save(embeddings, embedding_path)

    logging.info(f"Embeddings saved successfully at {embedding_path}.")
    logging.info(f"First Embedding: z type: {type(embeddings[0][0])}, shape: {embeddings[0][0].shape}, y type: {type(embeddings[0][1])}, shape: {embeddings[0][1].shape if isinstance(embeddings[0][1], torch.Tensor) else 'N/A'}, adj_file_path type: {type(embeddings[0][2])}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root_path', '-r', type=str, default=os.getcwd(), help='Root path (default is current directory)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to the model')
    parser.add_argument('--fold_index', '-f', type=int, required=True, help='Fold index')
    parser.add_argument('--num_features', '-n', type=int, default=658, help='Number of features (default is 658)')
    parser.add_argument('--embedding_dir', '-e', type=str, default='embeddings/', help='Embedding directory (default is embeddings/)')
    parser.add_argument('--csv_path', '-c', type=str, required=True, help='Path to the csv file')

    parser_args = parser.parse_args()

    root_path = parser_args.root_path
    debug = parser_args.debug
    model_path = parser_args.model_path
    fold_index = parser_args.fold_index
    num_features = parser_args.num_features
    embedding_dir = parser_args.embedding_dir
    csv_path = parser_args.csv_path

    setup_logging(debug)

    logging.info(f"Applying save_embeddings with parameters: root_path: {root_path}, model_path: {model_path}, fold_index: {fold_index}, num_features: {num_features}, embedding_dir: {embedding_dir}, csv_path: {csv_path}")

    df = pd.read_csv(csv_path)
    folds = list(df.groupby('fold'))
    fold_number, fold=folds[fold_index]
    train_fold = fold[fold["fold_label"] == "train"]
    test_fold = fold[fold["fold_label"] == "test"]
    support_fold = fold[fold["fold_label"] == "support"]

    logging.info(f"CSV file is read from {csv_path}. Number of folds: {len(folds)}. Fold {fold_index} is selected. Starting to save embeddings.")

    save_embeddings(root_path, model_path, train_fold, num_features, embedding_dir)

    logging.info("Train embeddings saved.")

    save_embeddings(root_path, model_path, test_fold, num_features, embedding_dir)

    logging.info("Test embeddings saved.")

    save_embeddings(root_path, model_path, support_fold, num_features, embedding_dir)

    logging.info("Support embeddings saved.")