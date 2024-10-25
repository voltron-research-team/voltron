import os
import warnings
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def sum_up_file_content(input_dir, critical_apis):
    line_count_dict = {}  # Dictionary to store line counts

    # Iterate over all text files in the folder
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            try:
                with open(file_path, "r") as file:
                    for line in file:
                        line = line.strip()
                        if line in line_count_dict:
                            line_count_dict[line]["Usage Count"] += 1
                        else:
                            is_dangerous = "Yes" if line in critical_apis else "No"
                            line_count_dict[line] = {"Usage Count": 1, "Dangerous API?": is_dangerous}
            except Exception as e:
                print(f"Exception in sum_up_file_content: {e}")

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(list(line_count_dict.values()))
    df.index = list(line_count_dict.keys())  # Set the line as the index
    df.index.name = "Line"
    df = df.reset_index()

    # Sort the DataFrame by the "Usage Count" column
    df = df.sort_values(by="Usage Count", ascending=False)

    return df


def merge_bw_and_mw_android_apis(benignware_df, malware_df):
    # Merge the two DataFrames on the 'Line' column
    merged_df = pd.merge(
        benignware_df[['Line', 'Usage Count', 'Dangerous API?']],
        malware_df[['Line', 'Usage Count']],
        on='Line',
        how='outer',
        suffixes=('_BW', '_MW')
    )

    # Fill NaN values with 0 for usage counts
    merged_df['Usage Count_BW'].fillna(0, inplace=True)
    merged_df['Usage Count_MW'].fillna(0, inplace=True)

    # If Dangerous API is NaN, fill it with 'No'
    merged_df['Dangerous API?'].fillna('No', inplace=True)

    # Rename columns to desired names
    merged_df.rename(columns={
        'Usage Count_BW': 'BW Usage Count',
        'Usage Count_MW': 'MW Usage Count'
    }, inplace=True)

    # Add Total Usage Count column
    merged_df['Total Usage Count'] = merged_df['BW Usage Count'] + merged_df['MW Usage Count']

    # Reorder columns
    merged_df = merged_df[['Line', 'BW Usage Count', 'MW Usage Count', 'Total Usage Count', 'Dangerous API?']]

    return merged_df


def choose_api_list(merged_df, output_txt_file, output_dir):
    # Select rows where 'Dangerous API?' is 'Yes'
    dangerous_apis = merged_df[merged_df['Dangerous API?'] == 'Yes']

    # Select rows where 'Total Usage Count' is greater than or equal to 20
    #dangerous_apis = dangerous_apis[dangerous_apis['Total Usage Count'] >= 20]

    # Filter out rows where 'MW Usage Count' is 0 or 1
    dangerous_apis = dangerous_apis[(dangerous_apis['MW Usage Count'] > 1)]

    # Select the 'Line' column and sort it in alphabetical order
    api_list = dangerous_apis['Line'].sort_values().tolist()

    # Create a dictionary with the API as key and its index as value
    api_dict = {api: idx for idx, api in enumerate(api_list)}

    # Write the dictionary to a text file
    with open(output_txt_file, 'w') as f:
        for api, idx in api_dict.items():
            f.write(f"{api}: {idx}\n")

    output_file = os.path.join(output_dir, "selected_apis_list.txt")

    try:
        with open(output_file, 'w') as f:
            for item in api_list:
                f.write(f"{item}\n")
        print(f"{output_file} is created successfully.")
    except Exception as e:
        print(f"Exception in writing selected APIs list: {e}")

    print(f"{output_txt_file} is created successfully.")

def write_into_excel(df, output_excel_file):

    # Create a new workbook and add the DataFrame to a worksheet
    wb = Workbook()
    ws = wb.active
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)

    # Add filters to the worksheet
    ws.auto_filter.ref = ws.dimensions

    # Adjust column widths
    ws.column_dimensions['A'].width = 80
    ws.column_dimensions['B'].width = 20
    ws.column_dimensions['C'].width = 20
    ws.column_dimensions['D'].width = 20
    ws.column_dimensions['E'].width = 20

    # Save the workbook to an Excel file
    wb.save(output_excel_file)

    print(f"{output_excel_file} is created successfully.")

def write_into_csv(df, output_csv_file):
    try:
        df.to_csv(output_csv_file, index=False)
        print(f"{output_csv_file} is created successfully.")
    except Exception as e:
        print(f"Exception in write_into_csv: {e}")

