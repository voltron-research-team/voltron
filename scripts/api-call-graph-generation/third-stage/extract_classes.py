import os
import re

method_path_bw_test = "D:/Thesis_outputs/FirstStage/4.custom_methods/bw_test/"
class_path_bw_test = "D:/Thesis_outputs/FirstStage/6.custom_classes/bw_test/"
method_path_bw_train = "D:/Thesis_outputs/FirstStage/4.custom_methods/bw_train/"
class_path_bw_train = "D:/Thesis_outputs/FirstStage/6.custom_classes/bw_train/"
method_path_bw_yedek = "D:/Thesis_outputs/FirstStage/4.custom_methods/bw_yedek/"
class_path_bw_yedek = "D:/Thesis_outputs/FirstStage/6.custom_classes/bw_yedek/"
method_path_mw_train = "D:/Thesis_outputs/FirstStage/4.custom_methods/mw_train/"
class_path_mw_train = "D:/Thesis_outputs/FirstStage/6.custom_classes/mw_train/"
method_path_mw_test = "D:/Thesis_outputs/FirstStage/4.custom_methods/mw_test/"
class_path_mw_test = "D:/Thesis_outputs/FirstStage/6.custom_classes/mw_test/"


def convert_method_to_class(input_path, output_path):
    try:
        os.makedirs(output_path, exist_ok=True)  # Create the output directory if it doesn't exist
        for file_name in os.listdir(input_path):
            if file_name.endswith('.txt'):
                input_file_path = os.path.join(input_path, file_name)
                output_file_path = os.path.join(output_path, file_name)

                with open(input_file_path, 'r', encoding='utf-8') as input_file:
                    method_names = input_file.readlines()

                # Extract unique class names using a set to eliminate duplicates
                unique_class_names = set(re.search(r'(.+);->', method).group(1) + "'" for method in method_names)

                # Sort the class names alphabetically
                sorted_class_names = sorted(unique_class_names)

                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.writelines('\n'.join(sorted_class_names))

                print(f"Conversion complete for {file_name}")
    except Exception as e:
        print(f"Error: {e}")


convert_method_to_class(method_path_mw_train, class_path_mw_train)
