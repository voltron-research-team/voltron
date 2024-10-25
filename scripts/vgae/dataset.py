import os
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import args
import pandas as pd
import logging
import io
class MultiGraphDataset(Dataset):
    def __init__(self, root, fold=None, preload=False, transform=None, pre_transform=None, force_correct_path=False):
        try:            
            self.root = root
            self.adj_files = []
            self.targets = []
            self.preload = preload

            if fold is None:
                logging.debug("No fold is found. Searching benign and malware subfolders.")
                for subfolder in [args.benignware_subfoler, args.malware_subfolder]:
                    subfolder_path = os.path.join(root, subfolder)
                    if os.path.isdir(subfolder_path):
                        for file in os.listdir(subfolder_path):
                            if file.endswith('.npy'):
                                self.adj_files.append(os.path.join(subfolder_path, file))
                                self.targets.append(0 if subfolder == args.benignware_subfoler else 1)
                
            else:
                logging.debug("Dataset Creation from Fold Started.")
                self.adj_files = fold['file_path'].apply(lambda x: os.path.normpath(os.path.join(root, x))).tolist()
                logging.debug(f"Adjacency files are created from fold. Number of files: {len(self.adj_files)}")
                self.targets = fold['type'].apply(lambda x: 1 if x == 'malware' else 0).tolist()
                logging.debug(f"Targets are created from fold. Number of targets: {len(self.targets)}")

                # Check if the files exist, if not delete the entry from the both list
                if force_correct_path:
                    logging.info(f"Force Correct Path is enabled. If the file does not exist, it will be removed from the list. Adjacency files: {len(self.adj_files)}. Targets: {len(self.targets)}")
                    for i in range(len(self.adj_files)-1, -1, -1):
                        if not os.path.exists(self.adj_files[i]):
                            logging.warning(f"File {self.adj_files[i]} does not exist. Removing from the list. Target: {self.targets[i]}")
                            del self.adj_files[i]
                            del self.targets[i]

            if self.preload:
                logging.debug("Preloading Started.")
                self.preloaded_data = [np.load(file) for file in self.adj_files]
                logging.debug("Preloading is done.")
            else:
                logging.debug("No Preloading. Data will be loaded when requested.")
                self.preloaded_data = None

            logging.info(f"Dataset is created from {root if fold is None else 'fold'}. Number of samples: {len(self.adj_files)}. Number of targets: {len(self.targets)}. Preload: {self.preload}")
        
            super(MultiGraphDataset, self).__init__(root, transform, pre_transform)
        except Exception as e:
            logging.error(f"An exception occurred when creating dataset: {e}", exc_info=True)
            raise e

    def len(self):
        return len(self.adj_files)

    def get(self, idx):
        try:            
            if self.preload:
                adj_matrix = self.preloaded_data[idx]
            else:
                with open(self.adj_files[idx], 'rb') as f:
                    try:
                        adj_matrix = np.load(f, allow_pickle=True)
                    except io.UnsupportedOperation as e:
                        raise Exception(f"Unsupported operation error occured at {f.name}: {e}")
                    except Exception as e:
                        raise Exception(f"An exception occurred when loading adjacency matrix from file {f.name}: {e}")
            
            # Create edge index from adjacency matrix
            edge_index = self.adj_to_edge_index(adj_matrix)

            # Create node features (one-hot encoding of node indices for simplicity)
            num_nodes = adj_matrix.shape[0]
            x = torch.eye(num_nodes)

            # Get target from the precomputed list
            target = torch.tensor([self.targets[idx]], dtype=torch.long)

        # Create Data object
            data = Data(
                x=x, 
                edge_index=edge_index,
                y=target
            )

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            logging.debug(f"Data is ready for index {idx}. Number of nodes: {num_nodes}. Number of edges: {edge_index.shape[1]}")
            return data
        except Exception as e:
            logging.error(f"An exception occurred when getting data: {e}", exc_info=True)
            raise e

    def adj_to_edge_index(self, adj_matrix):
        edge_index = np.array(np.nonzero(adj_matrix))
        return torch.tensor(edge_index, dtype=torch.long)


def read_fold(csv_path, fold_index):
    try:
        df = pd.read_csv(csv_path)
        folds = list(df.groupby('fold'))
        fold=folds[fold_index][1]
        train_fold = fold[fold["fold_label"] == "train"]
        test_fold = fold[fold["fold_label"] == "test"]

        logging.info(f"CSV file is read from {csv_path}. Number of folds: {len(folds)}. Fold {fold_index} is selected.")
        logging.debug(f"Train and test fold has been created with size: Train: {len(train_fold)} and Test: {len(test_fold)}")
        return train_fold, test_fold
    except Exception as e:
        logging.error(f"An exception occurred when reading CSV file: {e}", exc_info=True)
        raise e