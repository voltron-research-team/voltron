import json
import warnings
import sys
from first_stage_functions import create_file_and_graph
import os
import logging

def setup_logging(debug=False):
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(sys.stdout)
                        ])

def main(apk_dir, apk_group, output_dir, config_path):
    try:
        # Disable the user warnings
        warnings.filterwarnings("ignore", category=UserWarning, module=".*")

        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Access the file paths from the configuration
        # apk_info_output_dir = config["apk_info_output_dir"]
        # df_normalized_csv_dir = config["df_normalized_csv_dir"]
        # custom_methods_and_called_apis_output_dir = config["custom_methods_and_called_apis_output_dir"]
        # custom_methods_output_dir = config["custom_methods_output_dir"]
        # android_apis_output_dir = config["android_apis_output_dir"]

        apk_info_output_dir = os.path.join(output_dir, config["apk_info_output_dir"])
        df_normalized_csv_dir = os.path.join(output_dir, config["df_normalized_csv_dir"])
        custom_methods_and_called_apis_output_dir = os.path.join(output_dir, config["custom_methods_and_called_apis_output_dir"])
        custom_methods_output_dir = os.path.join(output_dir, config["custom_methods_output_dir"])
        android_apis_output_dir = os.path.join(output_dir, config["android_apis_output_dir"])

        # Create the 
        
        create_file_and_graph(apk_dir_or_file=apk_dir, 
                            apk_group=apk_group, 
                            apk_info_output_dir=apk_info_output_dir, 
                            custom_method_output_dir=custom_methods_output_dir, 
                            android_apis_output_dir=android_apis_output_dir, 
                            custom_methods_and_called_apis_output_dir=custom_methods_and_called_apis_output_dir, 
                            df_normalized_csv_dir=df_normalized_csv_dir)
    except Exception as e:
        logging.error(f"Error in main: {e}", exc_info=True)
        sys.exit(1)

    

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python main.py <apk_dir> <apk_group> <output_dir> <config_path>")
        sys.exit(1)

    apk_dir = sys.argv[1]
    apk_group = sys.argv[2]
    output_dir = sys.argv[3]
    config_path = sys.argv[4]

    setup_logging()

    main(apk_dir, apk_group, output_dir, config_path)
