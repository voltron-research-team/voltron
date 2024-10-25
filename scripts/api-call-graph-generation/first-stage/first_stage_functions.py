import os
import pickle
import pandas as pd
import re
from androguard.misc import AnalyzeAPK
import logging

def find_remaining_apks(input_dir_whole, output_dir_existing, apk_group):
    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    if not os.path.exists(directory_whole_files):
        os.makedirs(directory_whole_files)

    whole_apk_set = set([file_name for file_name in os.listdir(directory_whole_files)])

    directory_existing_files = os.path.join(output_dir_existing, apk_group)
    if not os.path.exists(directory_existing_files):
        os.makedirs(directory_existing_files)

    existing_output_files = set()
    for output_file in os.listdir(directory_existing_files):
        if output_file.endswith(".txt"):
            existing_output_files.add(os.path.splitext(output_file)[0])

    remaining_apk_files = whole_apk_set - existing_output_files
    logging.info("Number of remaining apk files for",  apk_group[:-1], "is:\t", len(remaining_apk_files))
    return remaining_apk_files

def load_serialized_objects(pkl_file_path):
    with open(pkl_file_path, "rb") as f:
        apk_info, disassembled_instructions, android_api_set = pickle.load(f)
    return apk_info, disassembled_instructions, android_api_set

def find_remaining_apks(input_dir_whole, output_dir_existing, apk_group):
    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    if not os.path.exists(directory_whole_files):
        os.makedirs(directory_whole_files)

    whole_apk_set = set([file_name for file_name in os.listdir(directory_whole_files)])

    directory_existing_files = os.path.join(output_dir_existing, apk_group)
    if not os.path.exists(directory_existing_files):
        os.makedirs(directory_existing_files)

    existing_output_files = set()
    for output_file in os.listdir(directory_existing_files):
        if output_file.endswith(".txt"):
            existing_output_files.add(os.path.splitext(output_file)[0])

    remaining_apk_files = whole_apk_set - existing_output_files
    print("Number of remaining apk files for",  apk_group[:-1], "is:\t", len(remaining_apk_files))
    return remaining_apk_files

def find_apk_file(directory_or_file):
    """
    Finds an APK file in the given directory or verifies if the given file is an APK.
    Args:
        directory_or_file (str): The path to a directory or a file.
    Returns:
        str: The path to the APK file.
    Raises:
        ValueError: If the provided path does not exist.
        ValueError: If the provided path is not a directory and does not end with '.apk'.
        ValueError: If no APK file is found in the directory.
        ValueError: If the provided path is not a directory or a file.
    """

    if not os.path.exists(directory_or_file):
        raise ValueError(f"The provided path {directory_or_file} does not exist.")
    
    if os.path.isfile(directory_or_file):
        file = directory_or_file
        if file.endswith('.apk'):
            return file
        else:
            raise ValueError(f"The provided path {directory_or_file} is not a directory and does not end with '.apk'")
    
    elif os.path.isdir(directory_or_file):
        directory = directory_or_file
    
        for file in os.listdir(directory):
            if file.endswith('.apk'):
                return file

        raise ValueError(f"No apk file found in {directory}")    
    
    else:
        raise ValueError(f"The provided path {directory_or_file} is not a directory or a file.")

def create_file_and_graph(apk_dir_or_file, apk_group, apk_info_output_dir, custom_method_output_dir,
                          android_apis_output_dir, custom_methods_and_called_apis_output_dir, df_normalized_csv_dir):
    try:
        apk_file = find_apk_file(apk_dir_or_file)

        if not apk_file:
            logging.warning(f"No apk file found in {apk_dir_or_file}")
            return
        
        

        if os.path.isdir(apk_dir_or_file):
            apk_path = os.path.join(apk_dir_or_file, apk_file)
            name_txt_path = os.path.join(apk_dir_or_file, "name.txt")

            if not os.path.exists(name_txt_path):
                logging.warning(f"Skipping {apk_dir_or_file} as it does not contain name.txt")
                return
            
            with open(name_txt_path, "r") as f:
                apk_name = f.read().strip()
        elif os.path.isfile(apk_dir_or_file):
            apk_path = apk_dir_or_file
            apk_name = os.path.splitext(apk_file)[0]
        else:
            raise ValueError(f"The provided path {apk_dir_or_file} is not a directory or a file.")
        
        logging.info(f"Analyzing {apk_file} in {apk_dir_or_file}")

        apk_info_output_file, df_normalized_csv_file, custom_methods_and_called_apis_output_file, custom_method_set_output_file, android_apis_output_file = \
            get_all_paths(apk_group, apk_info_output_dir, custom_method_output_dir, android_apis_output_dir, custom_methods_and_called_apis_output_dir,
                          df_normalized_csv_dir, name=apk_name)

        # Check if the APK has already been analyzed
        if os.path.exists(apk_info_output_file) and os.path.exists(df_normalized_csv_file) and os.path.exists(custom_methods_and_called_apis_output_file) and os.path.exists(custom_method_set_output_file) and os.path.exists(android_apis_output_file):
            logging.info(f"Skipping {apk_file} in {apk_dir_or_file}: It has already been analyzed.")
            return
        
        if not all([os.path.exists(file_path) for file_path in [apk_info_output_file, df_normalized_csv_file, custom_methods_and_called_apis_output_file, custom_method_set_output_file, android_apis_output_file]]):
            for file_path in [apk_info_output_file, df_normalized_csv_file, custom_methods_and_called_apis_output_file, custom_method_set_output_file, android_apis_output_file]:
                if os.path.exists(file_path):
                    logging.info(f"Deleting {file_path} as it is incomplete. for {apk_dir_or_file}: {apk_file}")
                    os.remove(file_path)
        
        # Create empty files
        for file_path in [apk_info_output_file, df_normalized_csv_file, custom_methods_and_called_apis_output_file, custom_method_set_output_file, android_apis_output_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    pass

        try:
        # Analyze the apk file using AnalyzeAPK
            a, d, dx = AnalyzeAPK(apk_path)  # a: APK object, d: DalvikVMFormat object, dx: Analysis object
        except Exception as e:
            raise RuntimeError(f"Androguard Error in {apk_dir_or_file} of {apk_name}: {e}")
        
        # Get call graph information of apk using apk_info_generation functions
        info_apk(a, apk_info_output_file)
        df_disassembled, general_android_api_set = get_disassembled_instructions(dx, apk_name=apk_name)
        df_normalized = get_normalized_instructions(df_disassembled, df_normalized_csv_file)
        custom_methods_and_call_apis = get_called_apis_from_custom_methods(df_normalized, custom_methods_and_called_apis_output_file)
        get_api_set(custom_methods_and_call_apis, general_android_api_set, custom_method_set_output_file, android_apis_output_file)
    except Exception as e:
        logging.error(f"An error occurred while analyzing {apk_dir_or_file}: {e}", exc_info=True)

        # delete the empty files if an error occurs
        for file_path in [apk_info_output_file, df_normalized_csv_file, custom_methods_and_called_apis_output_file, custom_method_set_output_file, android_apis_output_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

        logging.info(f"Deleted the empty files for {apk_dir_or_file}: {apk_file}")
        
        raise e

def get_all_paths(apk_group, apk_info_output_dir, custom_method_set_output_dir, android_apis_output_dir,
                  custom_methods_and_called_apis_output_dir, df_normalized_csv_dir, name):
    try:
            # apk_group = malware
        # apk_group = benignware

        f1 = os.path.join(apk_info_output_dir, apk_group, name + '.txt')
        f2 = os.path.join(df_normalized_csv_dir, apk_group, name + '.csv')
        f3 = os.path.join(custom_methods_and_called_apis_output_dir, apk_group, name + '.csv')
        f4 = os.path.join(custom_method_set_output_dir, apk_group, name + '.csv')
        f5 = os.path.join(android_apis_output_dir, apk_group, name + '.txt')

        file_paths = [f1, f2, f3, f4, f5]

        for file_path in file_paths:
            dir_path = os.path.dirname(file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

        return f1, f2, f3, f4, f5
    except Exception as e:
        logging.error(f"An error occurred while getting all paths: {e}")
        raise e

def info_apk(a, output_file):
    try:
        with open(output_file, 'w', encoding="utf-8") as f:
            print("APK permissions: ", a.get_permissions(), file=f)
            print("APK activities:", a.get_activities(), file=f)
            print("Package name:", a.get_package(), file=f)
            print("App name:", a.get_app_name(), file=f)
            print("App icon:", a.get_app_icon(), file=f)
            print("Android version code: ", a.get_androidversion_code(), file=f)
            print("Android version name: ", a.get_androidversion_name(), file=f)
            print("Min SDK version: ", a.get_min_sdk_version(), file=f)
            print("Max SDK version: ", a.get_max_sdk_version(), file=f)
            print("Target SDK version: ", a.get_target_sdk_version(), file=f)
            print("Effective target SDK version: ", a.get_effective_target_sdk_version(), file=f)
            print("Android manifest file: ", a.get_android_manifest_axml().get_xml(), file=f)
            logging.info(f"APK Permissions: {a.get_permissions()}, APK Activities: {a.get_activities()}, Package Name: {a.get_package()}, App Name: {a.get_app_name()}, Android Version Code: {a.get_androidversion_code()}, Android Version Name: {a.get_androidversion_name()}, Min SDK Version: {a.get_min_sdk_version()}, Max SDK Version: {a.get_max_sdk_version()}, Target SDK Version: {a.get_target_sdk_version()}, Effective Target SDK Version: {a.get_effective_target_sdk_version()}")
    except Exception as e:
        logging.error(f"Exception while getting APK info: {e}", exc_info=True)

def get_disassembled_instructions(dx, apk_name):
    try:
        data = []

        android_api_set = set()

        # check if dx has methods

        if not dx.get_methods():
            raise ValueError("The dx object does not have any methods.")
        
        
        for method in dx.get_methods():
            try:
                m = method.get_method()

                if method.is_android_api():
                    class_name = m.get_class_name()
                    method_name = m.get_name()
                    api_call = f"'{class_name}->{method_name}'"
                    android_api_set.add(api_call)

                if method.is_external():
                    continue

                # List of class name prefixes to be excluded or ignored
                excluded_prefixes = [
                    ["Landroid"],
                    ["Lcom", "google"],
                    ["Lcom", "android"],
                    ["Lkotlin"],
                    ["Lio", "flutter"]
                ]

                # Example class name
                class_name = m.get_class_name()

                # Split the class name by '/'
                split_class_name = class_name.split('/')

                def is_excluded(class_name_parts, prefixes):
                    for prefix in prefixes:
                        if class_name_parts[:len(prefix)] == prefix:
                            return True
                    return False

                logging.debug(f"Found any method {not is_excluded(split_class_name, excluded_prefixes)} in {class_name} in {apk_name}")
                
                if not is_excluded(split_class_name, excluded_prefixes):
                    for idx, ins in m.get_instructions_idx():
                        op_value = ins.get_op_value()
                        opcode = ins.get_name()
                        op_output = ins.get_output()

                        row_data = {
                            'method_name': m,
                            'idx': idx,
                            'op_value': op_value,
                            'opcode': opcode,
                            'opcode_output': op_output,
                        }

                        data.append(row_data)

                        logging.debug(f"Method: {m}, Index: {idx}, Opcode: {opcode}, Opcode Output: {op_output}, APK Name: {apk_name}")
            except Exception as e:
                logging.error(f"Exception while disassembling methods {method}: {e}", exc_info=True)
                continue
            
        df = pd.DataFrame(data)

        # check if the DataFrame is empty
        if df.empty:
            logging.warning("The DataFrame is empty.")
        
        # check if the DataFrame has the necessary columns
        required_columns = ['method_name', 'idx', 'op_value', 'opcode', 'opcode_output']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"DataFrame does not have a column named '{column}'. Colums of the DataFrame are: {df.columns}")

        return df, android_api_set
    except Exception as e:
        logging.error(f"Exception while disassembling instructions: {e}", exc_info=True)


        raise e

def get_normalized_instructions(df, file):
    try:        
        # Check if DataFrame has necessary columns
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input 'df' must be a pandas DataFrame.")
        
        required_columns = ['op_value', 'opcode_output', 'idx']
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"DataFrame does not have a column named '{column}'.")

        # Check if 'op_value' column contains numerical data
        if not pd.api.types.is_numeric_dtype(df['op_value']):
            raise TypeError("The 'op_value' column must contain numerical values.")
        
        # Check if 'file' is a valid string
        if not isinstance(file, str) or not file:
            raise ValueError("The 'file' parameter must be a non-empty string.")
    
        return_op_value_ranges = [(14, 17)]
        invoke_op_value_ranges = [(110, 120)]
        goto_switch_if_else_op_value_ranges = [(40, 44), (50, 61)]
        op_value_ranges = return_op_value_ranges + invoke_op_value_ranges + goto_switch_if_else_op_value_ranges

        op_value_mask = [not any(start <= val <= end for start, end in op_value_ranges) for val in df['op_value']]
        df = df.drop(df[op_value_mask].index)

        clone_pattern = r'[BCDFIJSZ]->clone()'
        call_pattern = r'L[^\(]*'

        for start, end in return_op_value_ranges:
            condition = (df['op_value'] >= start) & (df['op_value'] <= end)
            df.loc[condition, 'opcode_output'] = ""

        for start, end in invoke_op_value_ranges:
            condition = (df['op_value'] >= start) & (df['op_value'] <= end)
            df.loc[condition & df['opcode_output'].str.contains(clone_pattern, regex=True), 'opcode_output'] = 'Ljava/lang/Object;->clone'

        for start, end in invoke_op_value_ranges:
            condition = (df['op_value'] >= start) & (df['op_value'] <= end)
            df.loc[condition, 'opcode_output'] = df.loc[condition, 'opcode_output'].apply(lambda x: f"'{re.search(call_pattern, x).group(0)}'" if re.search(call_pattern, x) else x)

        hexadecimal_pattern = r'[-+][\da-f]+h'
        for start, end in goto_switch_if_else_op_value_ranges:
            condition = (df['op_value'] >= start) & (df['op_value'] <= end)
            for idx, row in df[condition].iterrows():
                hex_numbers = re.findall(hexadecimal_pattern, row['opcode_output'])
                goto_addresses = []
                for hex_num in hex_numbers:
                    hex_val = re.sub('h$', '', hex_num)
                    decimal_val = 2 * int(hex_val, 16)
                    goto_address = row['idx'] + decimal_val
                    goto_addresses.append(goto_address)
                    df.at[idx, 'opcode_output'] = goto_addresses

        df.to_csv(file, index=False)
        return df
    except Exception as e:
        logging.error(f"Exception while normalizing instructions in {file}: {e}", exc_info=True)
        raise e

def get_called_apis_from_custom_methods(df_normalized, file):
    try:
        custom_methods_and_call_apis = df_normalized[(df_normalized['op_value'] >= 110) & (df_normalized['op_value'] <= 120)]
        custom_methods_and_call_apis = custom_methods_and_call_apis[['method_name', 'opcode_output']]
        custom_methods_and_call_apis.to_csv(file, index=False)
        return custom_methods_and_call_apis
    except Exception as e:
        logging.error(f"Exception while getting called APIs from custom methods in {file}: {e}")
        raise e
    

def get_api_set(custom_methods_and_called_apis, general_android_api_set, custom_method_set_output_file, android_apis_output_file):
    try:
        all_called_methods_and_apis_set = set(map(str, custom_methods_and_called_apis['method_name'].tolist() + custom_methods_and_called_apis['opcode_output'].tolist()))
        
        all_android_api_set = all_called_methods_and_apis_set.intersection(map(str, general_android_api_set))
        
        custom_method_set = all_called_methods_and_apis_set - all_android_api_set

        custom_method_set = sorted(list(custom_method_set))
        android_api_set = sorted(list(all_android_api_set))
        
        try:
            with open(custom_method_set_output_file, 'w', encoding="utf-8") as f1, open(android_apis_output_file, 'w', encoding="utf-8") as f2:
                f1.writelines('\n'.join(custom_method_set) + '\n')
                f2.writelines('\n'.join(android_api_set) + '\n')
                logging.info(f"Custom methods and android APIs saved to {custom_method_set_output_file} and {android_apis_output_file}")
        except Exception as e:
            logging.error("Exception while writing custom methods and android APIs to file: ", e)
            raise e
    except Exception as e:
        logging.error(f"Exception while getting API set: {e}")
        raise e
