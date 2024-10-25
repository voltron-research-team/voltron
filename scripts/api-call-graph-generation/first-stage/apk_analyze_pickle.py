import os
import pickle
import logging
import sys
import time
from androguard.misc import AnalyzeAPK
import pandas as pd

# Setup logging
logging.basicConfig(filename='apk_analysis.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def analyzeAPK_and_save_pickle(input_dir, output_dir):
    input_dir_whole = input_dir

    logging.info(f"Analyzing APKs in {input_dir_whole} and saving results to {output_dir}")

    # Ensure the output directories exist
    benign_output_dir = os.path.join(output_dir, "benign")
    malware_output_dir = os.path.join(output_dir, "malware")
    os.makedirs(benign_output_dir, exist_ok=True)
    os.makedirs(malware_output_dir, exist_ok=True)

    # Ensure the input directory exists
    if not os.path.exists(input_dir_whole):
        logging.error(f"Input directory does not exist: {input_dir_whole}")
        return

    directory_whole_files = os.path.join(input_dir_whole)

    # Find all subdirectories containing APK files
    apk_directories = []
    for root, dirs, files in os.walk(directory_whole_files):
        if any(file.endswith('.apk') for file in files) and "name.txt" in files:
            apk_directories.append(root)

    # Count the number of APKs to analyze
    total_apks = len(apk_directories)
    benign_apks = sum(1 for d in apk_directories if 'benign' in d.lower())
    malware_apks = sum(1 for d in apk_directories if 'malware' in d.lower())

    logging.info(f"Total APKs to analyze: {total_apks}. Benign APKs: {benign_apks}, Malware APKs: {malware_apks}")

    apk_count = {'benign': 0, 'malware': 0}
    start_time = time.time()

    # Process each APK directory
    for i, apk_dir in enumerate(apk_directories):
        try:
            logging.info(f"Analyzing APK in: {apk_dir}")

            # Find the .apk file within the directory
            apk_file = find_apk_file(apk_dir)
            if not apk_file:
                logging.warning(f"Skipping {apk_dir} as it does not contain any .apk file")
                continue

            apk_path = os.path.join(apk_dir, apk_file)
            name_txt_path = os.path.join(apk_dir, "name.txt")

            if not os.path.exists(name_txt_path):
                logging.warning(f"Skipping {apk_dir} as it does not contain name.txt")
                continue

            # Read the name from name.txt
            with open(name_txt_path, "r") as f:
                apk_name = f.read().strip()

            # Determine the output file path based on benign or malware
            if 'malware' in apk_dir.lower():
                output_file = os.path.join(malware_output_dir, f"{apk_name}.pkl")
            else:
                output_file = os.path.join(benign_output_dir, f"{apk_name}.pkl")

            # Check if the APK has already been analyzed
            if os.path.exists(output_file):
                logging.info(f"Skipping {apk_dir} as it has already been analyzed.")
                continue

            # Start time for analyzing this APK
            apk_start_time = time.time()

            # Analyze the APK file using AnalyzeAPK
            a, d, dx = AnalyzeAPK(apk_path)  # a: APK object, d: DalvikVMFormat object, dx: Analysis object

            a_info = {
                "permissions": a.get_permissions(),
                "activities": a.get_activities(),
                "package_name": a.get_package(),
                "app_name": a.get_app_name(),
                "app_icon": a.get_app_icon(),
                "android_version_code": a.get_androidversion_code(),
                "android_version_name": a.get_androidversion_name(),
                "min_sdk_version": a.get_min_sdk_version(),
                "max_sdk_version": a.get_max_sdk_version(),
                "target_sdk_version": a.get_target_sdk_version(),
                "effective_target_sdk_version": a.get_effective_target_sdk_version(),
                "android_manifest": a.get_android_manifest_axml().get_xml(),
            }

            # Extract disassembled instructions and Android API set
            disassembled_instructions, android_api_set = get_disassembled_instructions(dx)

            # Serialize the extracted information to a file
            with open(output_file, "wb") as f:
                pickle.dump((a_info, disassembled_instructions, android_api_set), f)

            # End time for analyzing this APK
            apk_end_time = time.time()

            # Classify APK based on folder name and increment count
            if 'malware' in apk_dir.lower():
                apk_count['malware'] += 1
            else:
                apk_count['benign'] += 1

            # Calculate elapsed time and estimate remaining time
            elapsed_time = apk_end_time - start_time
            average_time_per_apk = elapsed_time / (i + 1)
            remaining_apks = total_apks - (i + 1)
            estimated_remaining_time = remaining_apks * average_time_per_apk

            logging.info(f"""APK and Analysis information of {apk_name} saved to {output_file}.\nTime taken for {apk_dir}: {apk_end_time - apk_start_time:.2f} seconds.\nEstimated remaining time: {estimated_remaining_time / 60:.2f} minutes.\nProgress: {i + 1}/{total_apks} APKs analyzed ({(i + 1) / total_apks * 100:.2f}%)""")

        except Exception as e:
            logging.error(f"Exception while analyzing {apk_dir}: {e}")

    logging.info(f"Total APKs analyzed - Benign: {apk_count['benign']}, Malware: {apk_count['malware']}")

def find_apk_file(directory):
    """ Find the first .apk file in a directory """
    for file in os.listdir(directory):
        if file.endswith('.apk'):
            return file
    return None

def get_disassembled_instructions(dx):
    data = []
    android_api_set = set()

    for method in dx.get_methods():
        m = method.get_method()

        if method.is_android_api():
            class_name = m.get_class_name()
            method_name = m.get_name()
            api_call = f"'{class_name}->{method_name}'"
            android_api_set.add(api_call)

        if method.is_external():
            continue

        # List of class name prefixes to be excluded or ignored
        excluded_prefixes = ["Landroid", "Ljava", "Lcom/google", "Lcom/android", "Lkotlin", "Lio/flutter"]

        if not any(m.get_class_name().startswith(prefix) for prefix in excluded_prefixes):
            for idx, ins in m.get_instructions_idx():
                
                if hasattr(ins, 'get_op_value'):
                    op_value = ins.get_op_value()
                else:
                    continue

                opcode = ins.get_name()
                op_output = ins.get_output()
                
                row_data = {
                    'method_name': m.get_name(),
                    'idx': idx,
                    'op_value': op_value,
                    'opcode': opcode,
                    'opcode_output': op_output,
                }
                
                data.append(row_data)

    df = pd.DataFrame(data)
    return df, android_api_set

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Call the function to analyze APKs and save results
    analyzeAPK_and_save_pickle(input_dir=input_dir, output_dir=output_dir)