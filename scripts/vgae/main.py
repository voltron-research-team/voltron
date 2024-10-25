from dataset import MultiGraphDataset
from logging_setup import setup_logging
from train import run_training
import argparse
import logging
from datetime import datetime
from test import run_testing
import os
import args
from signal_handler import register_signal_handlers
from folds import get_folds, check_fold_file_paths

logger = logging.getLogger()

register_signal_handlers()

def main(fold_info_csv_path, fold_files_root_path, test, preload, fold_index, checkpoint_path=None, force_correct_path=False):    
    train_fold, test_fold = get_folds(fold_index, fold_info_csv_path, test=test)

    if not check_fold_file_paths(train_fold, fold_files_root_path):
        raise ValueError("Train fold is None or invalid.")
    
    if test and not check_fold_file_paths(test_fold, fold_files_root_path):
        raise ValueError("Test fold is None or invalid.")
    
    train_dataset = MultiGraphDataset(root=fold_files_root_path, fold=train_fold, preload=preload, force_correct_path=force_correct_path)
    
    if checkpoint_path is None:
        run_training(train_dataset)
    else:
        run_training(train_dataset, checkpoint_path)
    
    if test:
        logger.info("Testing is started.")
        test_dataset = MultiGraphDataset(root=fold_files_root_path, fold=test_fold, preload=preload, force_correct_path=force_correct_path)
        run_testing(test_dataset, num_features=658)


if __name__ == '__main__':
    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get default values from args.py
        default_num_epochs = args.num_epochs
        default_batch_size = args.batch_size
        default_num_workers = args.num_workers
        default_learning_rate = args.learning_rate
        default_variational = args.variational
        default_visualize = args.visualize
        default_save = args.save
        default_model_name = args.model_name + "_" + current_time
        default_save_path = args.save_path
        default_save_period = args.save_period

        parser = argparse.ArgumentParser()
        
        parser.add_argument('--csv_path', '-c', type=str, required=True, help='Path to the csv file')
        parser.add_argument('--root_path', '-r', type=str, default=os.getcwd(), help='Root path (default is current directory)')
        parser.add_argument('--fold_index', '-f', type=int, default=0, help='Fold index (default is 0)')
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        parser.add_argument('--test', '-t', dest='test', action='store_true', default=True, help='Enable testing (default is True)')
        parser.add_argument('--no_test', '-nt', dest='test', action='store_false', help='Disable testing')
        parser.add_argument('--preload', '-p', action='store_true', default=False, help='Preload dataset (default is False)')
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
        parser.add_argument('--log_file', '-lf', type=str, default=None, help=f'Log file path (default is null)')
        parser.add_argument('--save_period', '-spr', type=int, default=default_save_period, help=f'Save period for model (default is {default_save_period})')
        parser.add_argument('--force_correct_path', '-fcp', action='store_true', help='Force correct path for the csv file')
        
        parser_args = parser.parse_args()
        
        csv_path = parser_args.csv_path
        debug = parser_args.debug
        test = parser_args.test
        root_path = parser_args.root_path
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
        fold_index = parser_args.fold_index
        log_file = parser_args.log_file
        save_period = parser_args.save_period
        force_correct_path = parser_args.force_correct_path

        args.batch_size = batch_size
        args.num_workers = num_workers
        args.learning_rate = learning_rate
        args.num_epochs = num_epochs
        args.variational = variational
        args.visualize = visualize
        args.save = save_model
        args.model_name = model_name
        args.save_path = save_path
        args.save_period = save_period

        setup_logging(debug, log_file)

        logging.info(f"""Application started with the following parameters:
                 csv_path: {csv_path},
                 debug: {debug},
                 test: {test},
                 root_path: {root_path},
                 preload: {preload},
                 save_model: {save_model},
                 save_path: {save_path},
                 save_period: {save_period},
                 model_name: {model_name},
                 batch_size: {batch_size},
                 num_workers: {num_workers},
                 learning_rate: {learning_rate},
                 num_epochs: {num_epochs},
                 variational: {variational},
                 visualize: {visualize},
                 fold_index: {fold_index},
                 log_file: {log_file},
                 force_correct_path: {force_correct_path}
                 """)
        
        main(csv_path, root_path, test, fold_index=fold_index, preload=preload, force_correct_path=force_correct_path)
    except Exception as e:
        logging.error(f"An exception occurred in main: {e}", exc_info=True)
