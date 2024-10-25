import os
import pickle
import pandas as pd
import re

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

def create_files_and_graphs(apk_group, input_dir, apk_info_output_dir, custom_method_set_output_dir,
                            android_apis_output_dir, custom_methods_and_called_apis_output_dir, df_normalized_csv_dir):
    input_dir_whole = input_dir
    if not os.path.exists(input_dir_whole):
        os.makedirs(input_dir_whole)

    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    if not os.path.exists(directory_whole_files):
        os.makedirs(directory_whole_files)

    whole_apk_set = set([file_name for file_name in os.listdir(directory_whole_files)])

    for apk_file in whole_apk_set:
        try:
            print("Analyzing", apk_file)
            pkl_file_path = os.path.join(directory_whole_files, apk_file)
            apk_info, disassembled_instructions, android_api_set = load_serialized_objects(pkl_file_path)
            
            f1, f2, f3, f4, f5 = get_all_paths(apk_group, apk_info_output_dir, custom_method_set_output_dir, android_apis_output_dir, custom_methods_and_called_apis_output_dir, df_normalized_csv_dir, apk_file)

            df_normalized = get_normalized_instructions(disassembled_instructions, f2)
            custom_methods_and_call_apis = get_called_apis_from_custom_methods(df_normalized, f3)
            get_api_set(custom_methods_and_call_apis, android_api_set, f4, f5)
        except Exception as e:
            print("Exception: ", e)

def get_all_paths(apk_group, apk_info_output_dir, custom_method_set_output_dir, android_apis_output_dir,
                  custom_methods_and_called_apis_output_dir, df_normalized_csv_dir, apk_file):
    if os.path.sep in apk_file or '/' in apk_file or '\\' in apk_file:
        directory_name, file_name = os.path.split(apk_file)
    else:
        file_name = apk_file
    name, ext = os.path.splitext(file_name)

    if apk_group != "benignware/":
        apk_group = "malware/"

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

def info_apk(a, output_file):
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

def get_disassembled_instructions_from_dx_info(dx_info):
    data = []
    android_api_set = set()

    for method in dx_info:
        if method['is_android_api']:
            class_name = method['class_name']
            method_name = method['name']
            api_call = f"'{class_name}->{method_name}'"
            android_api_set.add(api_call)

        if method['is_external']:
            continue

        excluded_prefixes = ["Landroid", "Ljava", "Lcom/google", "Lcom/android", "Lkotlin", "Lio/flutter"]

        if not any(method['class_name'].startswith(prefix) for prefix in excluded_prefixes):
            for ins in method['instructions']:
                # Adding a check to ensure 'op_value' is present in the instruction
                if 'op_value' not in ins:
                    print(f"Missing 'op_value' in instruction: {ins}")
                    continue
                row_data = {
                    'method_name': method['name'],
                    'idx': ins['idx'],
                    'op_value': ins['op_value'],
                    'opcode': ins['opcode'],
                    'opcode_output': ins['opcode_output'],
                }
                data.append(row_data)

    df = pd.DataFrame(data)
    return df, android_api_set


def get_normalized_instructions(df, file):
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

def get_called_apis_from_custom_methods(df_normalized, file):
    custom_methods_and_call_apis = df_normalized[(df_normalized['op_value'] >= 110) & (df_normalized['op_value'] <= 120)]
    custom_methods_and_call_apis = custom_methods_and_call_apis[['method_name', 'opcode_output']]
    custom_methods_and_call_apis.to_csv(file, index=False)
    return custom_methods_and_call_apis

def get_api_set(custom_methods_and_called_apis, general_android_api_set, custom_method_set_output_file, android_apis_output_file):
    all_called_methods_and_apis_set = set(custom_methods_and_called_apis['method_name'].tolist() + custom_methods_and_called_apis['opcode_output'].tolist())
    all_android_api_set = all_called_methods_and_apis_set.intersection(general_android_api_set)
    custom_method_set = all_called_methods_and_apis_set - all_android_api_set

    custom_method_set = sorted(custom_method_set)
    android_api_set = sorted(all_android_api_set)
    with open(custom_method_set_output_file, 'w', encoding="utf-8") as f1, open(android_apis_output_file, 'w', encoding="utf-8") as f2:
        f1.writelines('\n'.join(custom_method_set))
        f2.writelines('\n'.join(android_api_set))
