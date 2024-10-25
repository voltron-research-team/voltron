import argparse
import logging
from logging_setup import setup_logging
import os
from test import run_testing
from dataset import MultiGraphDataset
import args
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        parser.add_argument('--dataset_path', '-dp', type=str, required=True, help='Root path of the dataset')
        parser.add_argument('--preload', '-p', action='store_true', default=True, help='Preload dataset (default is True)')
        parser.add_argument('--no_preload', '-np', action='store_false', dest='preload', help='Do not preload dataset')
        parser.add_argument('--model_name', '-m', type=str, required=True, help=f'Model name prefix for saving')
        parser.add_argument('--save_path', '-sp', type=str, required=True, help=f'Path to save the model')

        parser_args = parser.parse_args()
        debug = parser_args.debug
        preload = parser_args.preload
        model_name = parser_args.model_name
        save_path = parser_args.save_path
        dataset_root_path = parser_args.dataset_path

        args.model_name = model_name
        args.save_path = save_path

        setup_logging(debug)

        logging.info(f"Application started with the following parameters: debug: {debug}, preload: {preload}, model_name: {model_name}, save_path: {save_path}")

        logging.info("Testing is started.")
        test_path = os.path.join(dataset_root_path, 'test')
        test_dataset = MultiGraphDataset(root=test_path, preload=True)
        run_testing(test_dataset)

    except Exception as e:
        logging.error(f"An exception occurred in main: {e}", exc_info=True)
        raise e