from dataset import MultiGraphDataset
from train import run_training
import argparse
import logging
import pandas as pd
from logging_setup import setup_logging
from test import run_testing
import os
import args


def main(dataset_root_path, test, preload):    
    
    train_path = os.path.join(dataset_root_path, 'train')
    train_dataset = MultiGraphDataset(root=train_path, preload=preload)
    run_training(train_dataset)

    if test:
        logging.info("Testing is started.")
        test_path = os.path.join(dataset_root_path, 'test')
        test_dataset = MultiGraphDataset(root=test_path, preload=preload)
        run_testing(test_dataset)


if __name__ == '__main__':
    try:
        # Get default values from args.py
        default_num_epochs = args.num_epochs
        default_batch_size = args.batch_size
        default_num_workers = args.num_workers
        default_learning_rate = args.learning_rate
        default_variational = args.variational
        default_visualize = args.visualize
        default_save = args.save
        default_model_name = args.model_name
        default_save_path = args.save_path
        default_dataset_path = args.path

        parser = argparse.ArgumentParser()
        
        parser.add_argument('--dataset_path', '-dp', type=str, default=default_dataset_path, help=f'Root path (default is current {default_dataset_path})')
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        parser.add_argument('--test', '-t', dest='test', action='store_true', default=True, help='Enable testing (default is True)')
        parser.add_argument('--no_test', '-nt', dest='test', action='store_false', help='Disable testing')
        parser.add_argument('--preload', '-p', action='store_true', default=True, help='Preload dataset (default is True)')
        parser.add_argument('--no_preload', '-np', action='store_false', dest='preload', help='Do not preload dataset')
        parser.add_argument('--save_model', '-s', action='store_true', default=default_save, help=f'Save model after training (default is {default_save})')
        parser.add_argument('--no_save_model', '-ns', action='store_false', dest='save_model', help='Do not save model after training')
        parser.add_argument('--model_name', '-m', type=str, default=default_model_name, help=f'Model name prefix for saving (default is "{default_model_name}")')
        parser.add_argument('--save_path', '-sp', type=str, default=default_save_path, help=f'Path to save the model (default is "{default_save_path}")')
        parser.add_argument('--batch_size', '-b', type=int, default=default_batch_size, help=f'Batch size (default is {default_batch_size})')
        parser.add_argument('--num_workers', '-w', type=int, default=default_num_workers, help=f'Number of workers for DataLoader (default is {default_num_workers})')
        parser.add_argument('--learning_rate', '-lr', type=float, default=default_learning_rate, help=f'Learning rate (default is {default_learning_rate})')
        parser.add_argument('--num_epochs', '-e', type=int, default=default_num_epochs, help=f'Number of epochs (default is {default_num_epochs})')
        parser.add_argument('--variational', '-v', action='store_true', default=default_variational, dest='variational', help=f'Enable variational mode (default is {default_variational})')
        parser.add_argument('--no_variational', '-nv', action='store_false', dest='variational', help='Disable variational mode')
        parser.add_argument('--visualize', '-vis', action='store_true', default=default_visualize, dest='visualize', help=f'Visualize the latent space (default is {default_visualize})')
        parser.add_argument('--no_visualize', '-nvis', action='store_false', dest='visualize', help='Do not visualize the latent space')

        parser_args = parser.parse_args()
        
        debug = parser_args.debug
        test = parser_args.test
        dataset_path = parser_args.dataset_path
        preload = parser_args.preload
        save_model = parser_args.save_model
        model_name = parser_args.model_name
        save_path = parser_args.save_path
        batch_size = parser_args.batch_size
        num_workers = parser_args.num_workers
        learning_rate = parser_args.learning_rate
        num_epochs = parser_args.num_epochs
        variational = parser_args.variational
        visualize = parser_args.visualize

        args.batch_size = batch_size
        args.num_workers = num_workers
        args.learning_rate = learning_rate
        args.num_epochs = num_epochs
        args.variational = variational
        args.visualize = visualize
        args.save = save_model
        args.model_name = model_name
        args.save_path = save_path
        args.path = dataset_path

        setup_logging(debug)

        logging.info(f"""Application started with the following parameters:
                 debug: {debug},
                 test: {test},
                 datset_path: {dataset_path},
                 preload: {preload},
                 save_model: {save_model},
                 model_name: {model_name},
                 save_path: {save_path},
                 batch_size: {batch_size},
                 num_workers: {num_workers},
                 learning_rate: {learning_rate},
                 num_epochs: {num_epochs},
                 variational: {variational},
                 visualize: {visualize}""")
        main(dataset_path, test, preload=preload)
    except Exception as e:
        logging.error(f"An exception occurred in main: {e}", exc_info=True)
