import args
from model import MultiGraphVGAE, save_model, check_save_conditions
from visual import visualize_latent_space
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import signal_handler as sh
import logging
import time


def train_epoch(model, loader, optimizer, criterion, device):
    try:
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            recon_loss = model.recon_loss(z, data.edge_index)
            if args.variational:
                recon_loss = recon_loss + (1 / data.num_nodes) * model.kl_loss()

            # Classification loss
            logits = model.classify(z, data.batch)
            class_loss = criterion(logits, data.y)  # Assuming data.y contains the labels

            loss = recon_loss + class_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            logging.debug(
                f"Batch {batch_idx + 1}/{len(loader)} trained. "
                f"Reconstruction Loss: {recon_loss.item():.4f}, "
                f"Classification Loss: {class_loss.item():.4f}, "
                f"Total Loss: {loss.item():.4f}"
            )
        return total_loss / len(loader)
    except Exception as e:
        logging.error(f"An exception occurred when training model: {e}", exc_info=True)
        raise e

def train_loop(device, model, loader, optimizer, criterion, num_epochs, save=False, save_period=None, save_path=None):
    n = save_period

    for epoch in range(num_epochs):
        if sh.stop_training:
            logging.warning(f"Stopping training after current epoch: {epoch}")
            break

        logging.debug(f"Training Epoch: {epoch} started.")
        start_time = time.time()
        loss = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        end_time = time.time()
        epoch_duration = end_time - start_time
        logging.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Time: {epoch_duration:.2f}s')
        
        if save and epoch % n == 0:
            save_model(model, save_path, epoch=epoch)
            model.to(device)

def run_training(dataset, checkpoint_model_path=None, checkpoint_optimizer_path=None):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type != 'cuda':
            logging.warning("CUDA is not available. Training will be slow.")

        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        model = MultiGraphVGAE(num_features=dataset.num_features, model_name=args.model_name).to(device)
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        criterion = CrossEntropyLoss()

        if checkpoint_model_path is not None:
            model.load_state_dict(torch.load(checkpoint_model_path))
            logging.info(f"Model loaded from {checkpoint_model_path}")

        if checkpoint_optimizer_path is not None:
            optimizer.load_state_dict(torch.load(checkpoint_optimizer_path))
            logging.info(f"Optimizer loaded from {checkpoint_optimizer_path}")

        if args.save:
            check_save_conditions(model=model, models_dir=args.save_path)

        logging.info(f"Training started with the following parameters: device: {device}, batch_size: {args.batch_size}, learning_rate: {args.learning_rate}, num_epochs: {args.num_epochs}, variational: {args.variational}, model_name: {args.model_name}")

        train_loop(device, model, loader, optimizer, criterion, args.num_epochs, args.save, args.save_period, args.save_path)

        # Save the moel  
        if (args.save):
            save_model(model, args.save_path)

        # Visualisation
        if (args.visualize):
            visualize_latent_space(model, dataset, method='tsne', keyword='train', device=device)
            visualize_latent_space(model, dataset, method='pca', keyword='train', device=device)
        
        logging.info("Training is completed successfully.")
    except Exception as e:
        raise Exception(f"An exception occurred when running training: {e}")