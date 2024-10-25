import sys
import os
import pickle

if __name__ == "__main__":
    selected_apis_list_path = sys.argv[1]
    output_file_path = sys.argv[2]

    if not os.path.exists(selected_apis_list_path):
        print(f"Error: {selected_apis_list_path} does not exist.")
        sys.exit(1)

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    api_id = 0
    api_dict = {}
    try:
        with open(selected_apis_list_path, 'r') as selected_apis_list_file, open(output_file_path, 'wb') as output_file:
            for line in selected_apis_list_file:
                api_name = line.strip()
                api_dict[api_name] = api_id
                api_id += 1
            
            pickle.dump(api_dict, output_file)
        
    except Exception as e:
        print(f"Error in main function: {e}")
        raise
    
    print("API dictionary creation completed successfully.")
