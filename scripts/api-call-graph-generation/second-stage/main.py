import warnings
import os
import sys
from first_stage_functions import find_remaining_apks, create_files_and_graphs
from second_stage_functions import write_into_csv, sum_up_file_content, merge_bw_and_mw_android_apis, choose_api_list, ensure_directories_exist
from third_stage_functions import general_configurations, all_steps_for_api_cg_creation

def main(output_dir, android_apis_dir, critical_api_full_list):
    # Disable the user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    benignware_folder = 'benignware/'
    malware_folder = 'malware/'

    # Disable the user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    # Ensure the output directories exist
    ensure_directories_exist([output_dir, android_apis_dir])

    with open(critical_api_full_list, 'r') as f:
        critical_apis = set(f.read().splitlines())

    # Bw Android APIs
    bw_input_dir = os.path.join(android_apis_dir, benignware_folder)
    bw_output_file = os.path.join(output_dir, "bw_apis.csv")

    ensure_directories_exist([bw_input_dir])

    bw_df = sum_up_file_content(bw_input_dir, critical_apis)
    #write_into_excel(bw_df, bw_output_file)
    write_into_csv(bw_df, bw_output_file)

    # Mw Android APIs
    mw_input_dir = os.path.join(android_apis_dir, malware_folder)
    mw_output_file = os.path.join(output_dir, "mw_apis.csv")

    ensure_directories_exist([mw_input_dir])

    mw_df = sum_up_file_content(mw_input_dir, critical_apis)
    #write_into_excel(mw_df, mw_output_file)
    write_into_csv(mw_df, mw_output_file)

    # Merge Bw and Mw Android APIs
    merged_output_file = os.path.join(output_dir, "all_apis.csv")
    merged_df = merge_bw_and_mw_android_apis(bw_df, mw_df)
    #write_into_excel(merged_df, merged_output_file)
    write_into_csv(merged_df, merged_output_file)

    # Select APIs
    output_txt_file = os.path.join(output_dir, "selected_apis.txt")
    choose_api_list(merged_df, output_txt_file, output_dir)

if __name__ == "__main__":
    try:
        output_dir = sys.argv[1]
        android_apis_dir = sys.argv[2]
        critical_api_full_list = sys.argv[3]

        main(output_dir, android_apis_dir, critical_api_full_list)
    except IndexError:
        print("Usage: python main.py <output_dir> <android_apis_dir> <critical_api_full_list>")
        sys.exit(1)
    except Exception as e:
        print("An error occurred: %s" % e)
        sys.exit(1)
