import time
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from dataset import MultiGraphDataset
from model import MultiGraphVGAE

def test_num_workers_and_batch_size(model, dataset, device, max_workers=32, start_batch_size=32, max_batch_size=1024):
    results = []
    batch_size = start_batch_size
    while batch_size <= max_batch_size:
        num_workers = 0
        while num_workers <= max_workers:
            try:
                if num_workers == 0:
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                else:
                    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                start_time = time.time()
                for i, data in enumerate(loader):
                    if i >= 10:  # Test with a fixed number of batches to get a reasonable measurement
                        break
                    data = data.to(device)
                    model.encode(data.x, data.edge_index)  # Run a forward pass to check memory usage and time
                end_time = time.time()
                duration = end_time - start_time
                print(f"Batch size: {batch_size}, Num workers: {num_workers}, Duration: {duration:.2f}s")
                results.append((batch_size, num_workers, duration))
                if num_workers == 0:
                    num_workers = 2
                else:
                    num_workers *= 2
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"Batch size {batch_size}, Num workers {num_workers} is too large. Out of memory error encountered.")
                    continue
                else:
                    raise e
        batch_size *= 2  # Double the batch size for the next iteration

    # Find the best combination
    best_combination = min(results, key=lambda x: x[2])
    print(f"Best combination: Batch size: {best_combination[0]}, Num workers: {best_combination[1]}, Duration: {best_combination[2]:.2f}s")
    return best_combination

if __name__ == "__main__":
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    root = None # Set the root directory of the dataset
    csv_path = None # Set the path to the csv file
    df = pd.read_csv(csv_path)
    folds = list(df.groupby('fold'))
    fold = folds[0][1]
    train_fold = fold[fold["fold_label"] == "train"]
    dataset = MultiGraphDataset(root=root, fold=train_fold)

    # Initialize model
    model = MultiGraphVGAE(num_features=dataset.num_features, model_name="Model_cuda").to(device)

    # Test num_workers and batch_size
    best_combination = test_num_workers_and_batch_size(model, dataset, device)
