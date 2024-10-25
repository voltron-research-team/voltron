import warnings
import sys
from third_stage_functions import api_cg_creation
import pickle
import logging
import os

def load_api_dict(file_path):
    try:
        with open(file_path, 'rb') as file:
            api_dict = pickle.load(file)
            
        return api_dict
    except Exception as e:
        logging.error(f"Error loading dictionary: {e}")
        raise

def setup_logging():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout)
                        ])

def main(apk_name, apk_group, output_dir, first_stage_dir, second_stage_dir):
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    selected_apis_list_path = os.path.join(second_stage_dir, "selected_apis_list_dict.pkl")
    try:
        api_dict = load_api_dict(selected_apis_list_path)
        
        # Benignware graphs
        api_cg_creation(apk_name=apk_name, apk_group=apk_group, selected_api_dict=api_dict, json_ouput_dir=output_dir, first_stage_dir=first_stage_dir)
        logging.info("API call graph creation completed successfully.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    try:
        setup_logging()

        apk_name = sys.argv[1]
        apk_group = sys.argv[2]
        output_dir = sys.argv[3]
        first_stage_dir = sys.argv[4]
        second_stage_dir = sys.argv[5]

        main(apk_name, apk_group, output_dir, first_stage_dir, second_stage_dir)

    except IndexError:
        logging.error("Usage: python main.py <apk_name> <apk_group> <output_dir> <first_stage_dir> <second_stage_dir>")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Exception in main: {e}")
        sys.exit(1)
