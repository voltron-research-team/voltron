import logging
import argparse
import args
from logging_setup import setup_logging
from test import run_testing
from dataset import MultiGraphDataset
import pandas as pd
import os

if __name__ == '__main__':
    try:
        default_batch_size = args.batch_size
        default_num_workers = args.num_workers

        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        parser.add_argument('--model_path', '-mp', type=str, required=True, help='Path to the model')
        parser.add_argument('--fold_index', '-f', type=int, default=0, help='Fold index')
        parser.add_argument('--csv_path', '-c', type=str, required=True, help='Path to the csv file')
        parser.add_argument('--num_features', '-nf', type=int, required=True, help='Number of features')
        parser.add_argument('--batch_size', '-b', type=int, default=default_batch_size, help='Batch size')
        parser.add_argument('--num_workers', '-w', type=int, default=default_num_workers, help='Number of workers')
        parser.add_argument('--root_path', '-r', type=str, default=os.getcwd(), help='Root path (default is current directory)')
        parser.add_argument('--preload', '-p', action='store_true', default=True, help='Preload dataset (default is True)')
        parser.add_argument('--no_preload', '-np', action='store_false', dest='preload', help='Do not preload dataset')
        
        parser_args = parser.parse_args()
        
        debug = parser_args.debug
        model_path = parser_args.model_path
        fold_index = parser_args.fold_index
        csv_path = parser_args.csv_path
        num_features = parser_args.num_features
        batch_size = parser_args.batch_size
        num_workers = parser_args.num_workers
        root_path = parser_args.root_path
        preload = parser_args.preload

        args.batch_size = batch_size
        args.num_workers = num_workers

        setup_logging(debug)

        logging.info(f"Application started with the following parameters: debug: {debug}, model_path: {model_path}, fold_index: {fold_index}, csv_path: {csv_path}, num_features: {num_features}, batch_size: {batch_size}, num_workers: {num_workers}, root_path: {root_path}, preload: {preload}")

        df = pd.read_csv(csv_path)
        folds = list(df.groupby('fold'))
        fold=folds[fold_index][1]
        test_fold = fold[fold["fold_label"] == "test"]

        dataset = MultiGraphDataset(root=root_path, fold=test_fold, preload=preload)

        logging.info(f"Dataset is created from fold. Number of samples: {len(dataset.adj_files)}. Number of targets: {len(dataset.targets)}. Preload: {dataset.preload}")

        run_testing(dataset, num_features=num_features, model_path=model_path)

        logging.info("Test is completed successfully.")
    except Exception as e:
        logging.error(f"An exception occurred in main: {e}", exc_info=True)
        raise e