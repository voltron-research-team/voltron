import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import logging
from torch_geometric.data import DataLoader
from dataset import MultiGraphDataset, read_fold
from model import MultiGraphVGAE, load_model_from_path
import sys
import datetime



logger = logging.getLogger(__name__)

def setup_logging(debug=False):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout)
                        ])

# Visualization function
def visualize_latent_space(model, dataset, method='tsne', keyword='test', device='cpu'):
    model.eval()
    latent_vectors = []
    labels = []  # Assuming your dataset provides labels

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            latent_vectors.append(z.mean(dim=0).cpu().numpy())  # Using mean of node embeddings as graph embedding
            labels.append(data.y.item())  # Assuming data.y contains the class labels for the graphs

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    reduced_vecs = reducer.fit_transform(latent_vectors)

    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_vecs[:, 0], reduced_vecs[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Latent Space Visualization using {method.upper()}')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    figure_path = "./results/res_{model}_{method}_{keyword}_{datetime}".format(
        model=model.model_name,
        method=method,
        keyword=keyword,
        datetime=current_datetime
    )

    plt.savefig(figure_path)
    
    # plt.show()

    logger.info(f'Latent space visualization saved to ./results/res_{model.model_name}_{method}_{keyword}.png')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', '-p', type=str, default=os.getcwd(), help='Root path (default is current directory)')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Path to the model')
    parser.add_argument('--fold', '-f', type=int, required=True, help='Fold index')
    parser.add_argument('--num_features', '-n', type=int, default=658, help='Number of features (default is 658)')
    parser.add_argument('--csv_path', '-c', type=str, required=True, help='Path to the csv file')
    parser_args = parser.parse_args()

    num_features = parser_args.num_features
    model_path = parser_args.model_path
    path = parser_args.path
    fold = parser_args.fold
    csv_path = parser_args.csv_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Visilualizing latent space startded for model {model_path} and fold {fold}. Arguments: num_features={num_features}, path={path}, fold={fold}, device={device}, model_path={model_path}, csv_path={csv_path}')
    # Load the model
    model = MultiGraphVGAE(num_features).to(device)
    load_model_from_path(model, model_path)

    train_fold, test_fold = read_fold(csv_path=csv_path, fold_index=fold)
    # Load the dataset
    test_dataset = MultiGraphDataset(root=path, fold=test_fold)
    train_dataset = MultiGraphDataset(root=path, fold=train_fold)
    # Visualize the latent space
    visualize_latent_space(model, test_dataset, method='tsne', keyword='test', device=device)
    visualize_latent_space(model, test_dataset, method='pca', keyword='test', device=device)
    visualize_latent_space(model, train_dataset, method='tsne', keyword='train', device=device)
    visualize_latent_space(model, train_dataset, method='pca', keyword='train', device=device)

