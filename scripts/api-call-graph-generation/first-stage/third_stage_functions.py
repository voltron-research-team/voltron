import json

import pandas as pd
import networkx as nx
import ast
import math
import warnings
import os
import traceback
from second_stage_functions import ensure_directories_exist
import time

def general_configurations(config_file):
    # Disable the user warnings
    warnings.filterwarnings("ignore", category=UserWarning, module=".*")

    with open(config_file, "r") as f:
        config = json.load(f)

    df_normalized_dir = config["df_normalized_dir"]
    custom_method_dir = config["custom_method_dir"]
    android_api_dir = config["android_api_dir"]
    selected_apis_used_in_apks = config["selected_apis_used_in_apks"]
    api_call_graphs_json_dir = config["api_call_graphs_json_dir"]
    api_call_graph_type = config["api_call_graph_type"]
    selected_apis_dictionary_file_path = config["selected_apis_dictionary_file_path"]

    # List of directories to ensure they exist
    directories = [
        df_normalized_dir,
        custom_method_dir,
        android_api_dir,
        api_call_graphs_json_dir
    ]

# Create directories if they do not exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

    # Dictionary of selected APIs used in APKs
    api_id = 0
    api_dict = {}
    with open(selected_apis_used_in_apks, 'r') as file:
        for line in file:
            api_name = line.strip()
            api_dict[api_name] = api_id
            api_id += 1
    with open(selected_apis_dictionary_file_path, "w") as file:
        for key, value in api_dict.items():
            # Write each key-value pair on a separate line
            file.write(f"{key}: {value}\n")

    return df_normalized_dir, custom_method_dir, android_api_dir, selected_apis_used_in_apks, \
           api_call_graphs_json_dir, api_call_graph_type, api_dict


def find_remaining_apks(input_dir_whole, output_dir_existing, apk_group):
    # Whole apk set
    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    ensure_directories_exist([directory_whole_files])

    whole_apk_set = set([file_name[:-4] for file_name in os.listdir(directory_whole_files)])

    # Existing output files
    directory_existing_files = os.path.join(output_dir_existing, apk_group)
    ensure_directories_exist([directory_existing_files])

    existing_output_files = set()
    for output_file in os.listdir(directory_existing_files):
        if output_file.endswith(".json"):
            existing_output_files.add(os.path.splitext(output_file)[0])

    # Remaining APK files
    remaining_apk_files = whole_apk_set - existing_output_files
    print("Number of remaining apk files for", apk_group[:-1], "is:\t", len(remaining_apk_files))
    return remaining_apk_files


def all_steps_for_api_cg_creation(apk_group, remaining_apk_files, android_api_dir, custom_method_dir,
                                  df_normalized_dir, api_call_graphs_json_dir, api_call_graph_type, selected_api_dict,
                                  apk_type):

    # Analyze each APK file
    input_dir_whole = "android_apis"
    directory_whole_files = os.path.join(input_dir_whole, apk_group)
    remaining_apk_files = set([file_name[:-4] for file_name in os.listdir(directory_whole_files)])

    for apk_name in remaining_apk_files:
        try:
            start_time = time.time()

            print("Analyzing", apk_name)

            # Get Android APIs used in the APK
            android_api_path = os.path.join(android_api_dir, apk_group, apk_name + '.txt')
            with open(android_api_path, 'r') as f:
                android_apis = set(f.read().splitlines())
                print("number of Android APIs used in the APK:", len(android_apis))

            # Get selected Android APIs used in the APK
            dangerous_android_apis = android_apis.intersection(selected_api_dict)

            # Get custom methods used in the apk
            custom_method_path = os.path.join(custom_method_dir, apk_group, apk_name + '.csv')
            with open(custom_method_path, 'r', encoding='utf-8') as f:
                custom_methods = set(f.read().splitlines())
            print("number of custom_methods:", len(custom_methods))

            # Get df_normalized
            df_normalized_path = os.path.join(df_normalized_dir, apk_group, apk_name + '.csv')
            df_normalized = pd.read_csv(df_normalized_path)
            df_normalized['opcode_output'] = df_normalized['opcode_output'].apply(deserialize_opcode_output)

            # Generate control flow graph
            graph = generate_cfg(df_normalized, custom_methods, android_apis)

            # Create output file paths
            api_cg_path = os.path.join(api_call_graphs_json_dir, apk_group, apk_name + '.json')
            ensure_directories_exist([os.path.dirname(api_cg_path)])

            # Create API Call Graphs
            """if api_call_graph_type == "Small":
                get_api_call_graph(apk_type, graph, selected_api_dict, dangerous_android_apis, api_cg_path)
            elif api_call_graph_type == "Big":
                get_homogeneous_api_call_graph(apk_type, graph, selected_api_dict, dangerous_android_apis, api_cg_path)
        """
            get_api_call_graph(apk_type, graph, selected_api_dict, dangerous_android_apis, api_cg_path)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken for {apk_name}: {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"Exception: {e}")
            print("Detailed traceback:")
            print(traceback.format_exc())


def deserialize_opcode_output(cell_value):
    if isinstance(cell_value, float) and math.isnan(cell_value):
        return cell_value
    try:
        if cell_value.startswith('['):
            return ast.literal_eval(cell_value)
        else:
            return cell_value
    except (ValueError, SyntaxError):
        return cell_value


def generate_cfg(df, custom_method_set, android_api_set):
    # create a new graph with only the critical APIs and their edges
    graph = nx.DiGraph()

    # add nodes for custom method calls
    for api in custom_method_set:
        return_api = f"return-{api}"
        graph.add_node(api)
        graph.add_node(return_api)

    # add nodes for Android API calls used in the APK
    graph.add_nodes_from(android_api_set)

    grouped = df.groupby('method_name')
    for method_name, sub_df in grouped:
        graph = generate_subgraph(graph, method_name, sub_df, android_api_set, custom_method_set)

    # Find isolated nodes
    isolated_nodes = [node for node in graph if graph.degree(node) == 0]

    # Remove isolated nodes from the graph
    graph.remove_nodes_from(isolated_nodes)

    # write_graph_info(graph, textfile, graphml)

    return graph


def generate_subgraph(graph, current_method, sub_df, android_api_set, custom_method_set):
    return_op_value_ranges = {(14, 17)}
    goto_op_value_ranges = {(40, 42)}
    if_else_switch_op_value_ranges = {(43, 44), (50, 61)}
    invoke_op_value_ranges = {(110, 120)}

    last_node = current_method
    return_method = f"return-{current_method}"
    for index, row in sub_df.iterrows():
        # "invoke" instructions
        if any(start <= row[2] <= end for start, end in invoke_op_value_ranges):
            api_call = row[4]
            if api_call in android_api_set:
                graph.add_edge(last_node, api_call)
                last_node = api_call
            elif api_call in custom_method_set:
                graph.add_edge(last_node, api_call)
                return_api_call = f"return-{api_call}"
                last_node = return_api_call

        # "return" instructions
        elif any(start <= row[2] <= end for start, end in return_op_value_ranges):
            graph.add_edge(last_node, return_method)

        # "if-else" and "switch" instructions
        elif any(start <= row[2] <= end for start, end in if_else_switch_op_value_ranges):
            branch_rows = []
            visited_rows = set()
            for row_index in row[4]:
                filtered_df = sub_df[sub_df['idx'] >= row_index]
                if not filtered_df.empty:
                    nearest_row = filtered_df.iloc[0]
                    if nearest_row[1] not in visited_rows:
                        visited_rows.add(nearest_row[1])
                        branch_rows.append(nearest_row)
            for branch_row in branch_rows:
                if any(start <= branch_row[2] <= end for start, end in invoke_op_value_ranges):
                    api_call = branch_row[4]
                    graph.add_edge(last_node, api_call)
                elif any(start <= branch_row[2] <= end for start, end in return_op_value_ranges):
                    graph.add_edge(last_node, return_method)
                elif any(start <= branch_row[2] <= end for start, end in if_else_switch_op_value_ranges):
                    filtered_next_row_add = sub_df[sub_df['idx'] > branch_row[1]]
                    if not filtered_next_row_add.empty:
                        next_row_add = filtered_next_row_add.iloc[0]
                        if next_row_add[1] not in visited_rows:
                            visited_rows.add(next_row_add[1])
                            branch_rows.append(next_row_add)
                    for branch_row_index in branch_row[4]:
                        filtered_branch_df = sub_df[sub_df['idx'] >= branch_row_index]
                        if not filtered_branch_df.empty:
                            nearest_row_add = filtered_branch_df.iloc[0]
                            if nearest_row_add[1] not in visited_rows:
                                visited_rows.add(nearest_row_add[1])
                                branch_rows.append(nearest_row_add)

        # "go-to" instructions
        elif any(start <= row[2] <= end for start, end in goto_op_value_ranges):
            goto_rows = []
            visited_goto = set()
            goto_row_index = row[4][0]
            filtered_df = sub_df[sub_df['idx'] >= goto_row_index]
            if not filtered_df.empty:
                goto_row = filtered_df.iloc[0]
                if goto_row[1] not in visited_goto:
                    visited_goto.add(goto_row[1])
                    goto_rows.append(goto_row)
            for goto_row in goto_rows:
                if any(start <= goto_row[2] <= end for start, end in invoke_op_value_ranges):
                    api_call = goto_row[4]
                    graph.add_edge(last_node, api_call)
                elif any(start <= goto_row[2] <= end for start, end in return_op_value_ranges):
                    graph.add_edge(last_node, return_method)
                elif any(start <= goto_row[2] <= end for start, end in goto_op_value_ranges):
                    new_goto_row_index = goto_row[4][0]
                    filtered_new_goto_row = sub_df[sub_df['idx'] >= new_goto_row_index]
                    if not filtered_new_goto_row.empty:
                        new_goto_row = filtered_new_goto_row.iloc[0]
                        if new_goto_row[1] not in visited_goto:
                            visited_goto.add(new_goto_row[1])
                            goto_rows.append(new_goto_row)
                elif any(start <= goto_row[2] <= end for start, end in if_else_switch_op_value_ranges):
                    current_row_index = goto_row[1]
                    filtered_next_row = sub_df[sub_df['idx'] > current_row_index]
                    if not filtered_next_row.empty:
                        next_row = filtered_next_row.iloc[0]
                        if next_row[1] not in visited_goto:
                            visited_goto.add(next_row[1])
                            goto_rows.append(next_row)
                    for row_index in goto_row[4]:
                        filtered_df = sub_df[sub_df['idx'] >= row_index]
                        if not filtered_df.empty:
                            nearest_row = filtered_df.iloc[0]
                            if nearest_row[1] not in visited_goto:
                                visited_goto.add(nearest_row[1])
                                goto_rows.append(nearest_row)

    return graph


def get_homogeneous_api_call_graph(apk_type, call_graph, given_api_dict, given_api_set, json_file):
    homogeneous_api_call_graph = nx.DiGraph()

    # set the graph-level attribute "is_malware"
    homogeneous_api_call_graph.graph['is_malware'] = apk_type  # 1 if malware, 0 if benignware

    # add nodes for critical Android API calls
    for api_name, api_id in given_api_dict.items():
        if api_name in given_api_set:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'red',
                          'is_used_in_program': 1}
        else:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'white',
                          'is_used_in_program': 0}
        homogeneous_api_call_graph.add_node(api_name, **attributes)

    # create edges
    for node in given_api_set:
        api_successors = set()
        visited = set()
        successor_list = []
        for neighbor in call_graph.successors(node):
            successor_list.append(neighbor)
        while successor_list:
            successor = successor_list[0]
            successor_list.remove(successor)
            if successor not in visited:
                visited.add(successor)
                if successor in given_api_set:
                    api_successors.add(successor)
                else:
                    new_successors = call_graph.successors(successor)
                    for item in new_successors:
                        successor_list.append(item)
        for target_api in api_successors:
            homogeneous_api_call_graph.add_edge(node, target_api)

    """
    # add self-edges to isolated nodes
    for node in nx.isolates(homogeneous_api_call_graph):
        homogeneous_api_call_graph.add_edge(node, node)
    """
    # relabel nodes with unique API ID
    homogeneous_critical_api_call_graph = nx.relabel_nodes(homogeneous_api_call_graph, given_api_dict)

    # write_graph_info(homogeneous_critical_api_call_graph, textfile, graphml)

    api_call_graph_json(apk_type, homogeneous_critical_api_call_graph, json_file)


def write_graph_info(graph, textfile, graphml):
    # create a list of dictionaries to store the edge information
    edge_info = []

    # edge information
    for i, edge in enumerate(graph.edges()):
        edge_info.append({
            'Edge Number': i,
            'Source Method': str(edge[0]),
            'Target Method': str(edge[1])
        })

    # create a pandas DataFrame from the edge information
    df = pd.DataFrame(edge_info)

    df.to_csv(textfile, sep='\t', index=False)
    """
    for u, v, edge_attrs in graph.edges(data=True):
        if 'color' not in graph.nodes[u]:
            print("u:", u)
        if 'color' not in graph.nodes[v]:
            print("v:", v)
        if graph.nodes[u]['color'] == 'red' or graph.nodes[v]['color'] == 'red':
            edge_attrs['color'] = 'red'
        else:
            edge_attrs['color'] = 'green'
    """
    # write graph into graphml file
    nx.write_graphml(graph, graphml)


def api_call_graph_json(apk_type, graph, json_file):
    # get the method names for the node labels
    node_labels = {}
    for i, node in enumerate(graph.nodes()):
        node_labels[i] = node

    # get edge information
    edges = []
    for source, target in graph.edges():
        edges.append([list(graph.nodes()).index(source), list(graph.nodes()).index(target)])

    # create dictionary to store data
    data = {"target": apk_type, "edges": edges, "labels": {}, "inverse_labels": {}}

    # add labels for each node
    for node, label in node_labels.items():
        data["labels"][str(node)] = label

    # add inverse labels
    for node, label in node_labels.items():
        if label not in data["inverse_labels"]:
            data["inverse_labels"][label] = []
        data["inverse_labels"][label].append(node)

    # write data to JSON file
    with open(json_file, "w") as f:
        json.dump(data, f)


def get_api_call_graph(apk_type, call_graph, given_api_dict, given_api_set, json_file):
    api_call_graph = nx.DiGraph()

    # set the graph-level attribute "is_malware"
    api_call_graph.graph['is_malware'] = apk_type  # 1 if malware, 0 if benignware

    # add nodes for critical Android API calls
    for api_name, api_id in given_api_dict.items():
        if api_name in given_api_set:
            attributes = {'api_name': api_name, 'api_id': api_id, 'color': 'red',
                          'is_used_in_program': 1}
            api_call_graph.add_node(api_name, **attributes)

    # create edges
    for node in given_api_set:
        api_successors = set()
        visited = set()
        successor_list = []
        if call_graph.has_node(node):
            for neighbor in call_graph.successors(node):
                successor_list.append(neighbor)
            while successor_list:
                successor = successor_list[0]
                successor_list.remove(successor)
                if successor not in visited:
                    visited.add(successor)
                    if successor in given_api_set:
                        api_successors.add(successor)
                    else:
                        new_successors = call_graph.successors(successor)
                        for item in new_successors:
                            successor_list.append(item)
            for target_api in api_successors:
                api_call_graph.add_edge(node, target_api)

    """
    # add self-edges to isolated nodes
    for node in nx.isolates(api_call_graph):
        api_call_graph.add_edge(node, node)
    """

    # relabel nodes with unique API ID
    critical_api_call_graph = nx.relabel_nodes(api_call_graph, given_api_dict)

    # write_graph_info(critical_api_call_graph, textfile, graphml)

    api_call_graph_json(apk_type, critical_api_call_graph, json_file)
