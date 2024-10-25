import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import TestDataset, FewShotDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import args
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib



def test_siamese_network(model, test_embeddings_path,support_set_path, criterion, device, support_set_size, batch_size=args.batch, threshold=args.threshold, num_workers=args.num_workers, save_results=False, few_shot=False):
    if few_shot:
        return test_siamese_network_few(model, test_embeddings_path,support_set_path, criterion, device, support_set_size, batch_size, threshold, num_workers, save_results)
    else:
        return test_siamese_network_zero(model, test_embeddings_path,support_set_path, criterion, device, support_set_size, batch_size, threshold, num_workers, save_results)
    

def test_siamese_network_zero(model, test_embeddings_path,support_set_path, criterion, device, support_set_size, batch_size=args.batch, threshold=args.threshold, num_workers=args.num_workers, save_results=False):
    test_embeddings = torch.load(test_embeddings_path, map_location=device)
    support_set = torch.load(support_set_path, map_location=device)
   
    # Create the test dataset and loader
    test_dataset = TestDataset(
        embeddings=test_embeddings, 
        support_set=support_set, 
        support_set_size=support_set_size)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate the model
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_paths = []
    
    with torch.no_grad():
        for embedding1, embedding2, label, path in test_loader:
            #embedding1, embedding2, label = embedding1.to(device), embedding2.to(device), label.to(device)

            # label is 1 if the embedding1 is malware, 0 otherwise
            
            output = sum([model(embedding1, embed) for embed in embedding2])/len(embedding2)
            loss = criterion(output, label)
            
            total_loss += loss.item()

            # Collect predictions and labels for metric calculations
            predictions = (output > threshold).float()

            # reverse the predictions
            predictions = 1 - predictions

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_paths.extend(path)
    
    average_loss = total_loss / len(test_loader)
    
    accuracy, precision, recall, f1, FPR, false_positives = get_metrics(all_labels, all_predictions)
    
    print(f'Zero Test Results: Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, FPR: {FPR[0]*100:.4f}%, False Positives: {false_positives}')
    
    if save_results:
        raw_df = save_results_to_csv(all_labels, all_predictions, all_paths, model.model_name)
        parsed_df = parse_path_and_create_df(raw_df, model.model_name)
        metrics_df = calculate_metrics_per_family(parsed_df, model.model_name)
        plot_metrics(metrics_df, model.model_name)

    return average_loss, accuracy, precision, recall, f1


def test_siamese_network_few(model, test_embeddings_path,support_set_path, criterion, device, shot_size, batch_size=args.batch, threshold=args.threshold, num_workers=args.num_workers, save_results=False):
    # Load the test embeddings
    test_embeddings = torch.load(test_embeddings_path, map_location=device)
    support_set = torch.load(support_set_path, map_location=device)
   
    # Create the test dataset and loader
    test_dataset = FewShotDataset(
        embeddings=test_embeddings, 
        support_set=support_set, 
        shot_size=5,
        benign_shot_size=20,
        familyName='SMSspy'
        )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate the model
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_paths = []
    
    with torch.no_grad():
        for embedding1, embedding2, embedding3, label, path in test_loader:

            benign_output = sum([model(embedding1, embed) for embed in embedding2]) / len(embedding2)
            malware_output = sum([model(embedding1, embed) for embed in embedding3]) / len(embedding3)

            predictions = (malware_output >= benign_output).float()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions)
            all_paths.extend(path)
    
    accuracy, precision, recall, f1, FPR, false_positives = get_metrics(all_labels, all_predictions)
    print(f'Few Shot Test Results: Test Accuracy: {accuracy:.4f}, '
          f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, FPR: {FPR[0]*100:.4f}%, False Positives: {false_positives}')
    
    if save_results:
        raw_df = save_results_to_csv(all_labels, all_predictions, all_paths, model.model_name)
        parsed_df = parse_path_and_create_df(raw_df, model.model_name)
        metrics_df = calculate_metrics_per_family(parsed_df, model.model_name)
        plot_metrics(metrics_df, model.model_name)

    return None, accuracy, precision, recall, f1


def get_metrics(all_labels, all_predictions):
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    false_positives = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
    
    # Calculate false positive rate (FPR)
    FP = sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
    TN = sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0))
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    return accuracy, precision, recall, f1, FPR, false_positives


def save_results_to_csv(all_labels, all_predictions, all_paths, model_name):
    # Determine the classification result
    classification = []
    for label, prediction in zip(all_labels, all_predictions):
        if label == 1 and prediction == 1:
            classification.append('TP')  # True Positive
        elif label == 0 and prediction == 0:
            classification.append('TN')  # True Negative
        elif label == 0 and prediction == 1:
            classification.append('FP')  # False Positive
        elif label == 1 and prediction == 0:
            classification.append('FN')  # False Negative
    
    all_labels = np.ravel(all_labels)  # Flatten to 1D if necessary
    all_labels = all_labels.astype(int)  # Convert to integers
    all_predictions = np.ravel(all_predictions)
    all_paths = np.ravel(all_paths)


    # Create the DataFrame
    results_df = pd.DataFrame({
        'Path': all_paths,
        'Label': all_labels,
        'Prediction': all_predictions.astype(int),
        'Correct': (all_labels == all_predictions).astype(int), # todo as int 0 or 1
        'Classification': classification
    })

    # Save to CSV
    directory = 'results'
    if not os.path.exists(directory):
        os.makedirs(directory)

    results_df.to_csv(f'{directory}/raw_results_{model_name}.csv', index=False)

    return results_df


def parse_path_and_create_df(raw_df, model_name):
    # Initialize lists to store extracted values
    package_names = []
    families = []

    for path in raw_df['Path']:                
        # Split the path to extract relevant components
        path_parts = path.split(os.sep)
        
        if 'malware_familied' in path_parts:
            family_name = path_parts[-2]  # Family name is the parent folder of the file
        elif 'benignware' in path_parts:
            family_name = None
        else:
            raise ValueError(f"Invalid path structure for path: {path}")
        
        # Extract package name (file name without extension)
        package_name = os.path.splitext(os.path.basename(path))[0]
        
        # Append extracted values to lists
        package_names.append(package_name)
        families.append(family_name)
    
    # Create a new DataFrame with the parsed information
    parsed_df = pd.DataFrame({
        'Path': raw_df['Path'],
        'Package': package_names,
        'Family': families, # None for benignware
        'Label': raw_df['Label'],
        'Prediction': raw_df['Prediction'],
        'Correct': raw_df['Correct'],
        'Classification': raw_df['Classification']
    })

    parsed_df.to_csv(f'results/parsed_results_{model_name}.csv', index=False)
    
    return parsed_df


def calculate_metrics_per_family(parsed_df, model_name):
    # Initialize a dictionary to store metrics
    metrics = {
        'Family': [],
        'Total': [],
        'TP': [],
        'FN': [],
        'Recall': [],
    }

    # Get unique families
    families = parsed_df['Family'].unique()

    for family in families:
        # Filter the DataFrame by family
        family_df = parsed_df[parsed_df['Family'] == family]
        
        if family_df.empty or family is None:
            continue

        # True Positives (TP): Correctly predicted as positive
        tp = len(family_df[(family_df['Label'] == 1) & (family_df['Prediction'] == 1)])
        
        # False Negatives (FN): Incorrectly predicted as negative
        fn = len(family_df[(family_df['Label'] == 1) & (family_df['Prediction'] == 0)])
        
        # False Positives (FP): Incorrectly predicted as positive
        fp = len(family_df[(family_df['Label'] == 0) & (family_df['Prediction'] == 1)])
        
        # True Negatives (TN): Correctly predicted as negative
        tn = len(family_df[(family_df['Label'] == 0) & (family_df['Prediction'] == 0)])
        
        # Calculate Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
        # Append the results to the dictionary
        metrics['Family'].append(family)
        metrics['TP'].append(tp)
        metrics['FN'].append(fn)
        metrics['Recall'].append(recall)
        metrics['Total'].append(tp + fn + fp + tn)

    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'results/metrics_mw_{model_name}.csv', index=False)
    return metrics_df


def plot_metrics(metrics_df, model_name=None):
    fig, ax1 = plt.subplots(figsize=(25, 8))

    # Sort by recall in descending order
    metrics_df = metrics_df.sort_values(by='Recall', ascending=False)

    # Plot True Positives (TP) as a proportion of Total
    tp_proportion = metrics_df['TP'] / metrics_df['Total']
    fn_proportion = metrics_df['FN'] / metrics_df['Total']

    bar_width = 0.7
    bar1 = ax1.bar(metrics_df['Family'], tp_proportion, color='#1f77b4', alpha=0.8, label='True Predictions', width=bar_width)

    # Overlay False Negatives (FN) in orange on the same bars
    bar2 = ax1.bar(metrics_df['Family'], fn_proportion, bottom=tp_proportion, color='#ff7f0e', alpha=0.8, label='False Predictions', width=bar_width)

    ax1.set_xlabel('Family')
    ax1.set_ylabel('Proportion', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    # Title and Legend
    plt.title('True vs False Predictions per Family', fontsize=16)
    ax1.legend(loc='upper right')

    for i, rect in enumerate(bar1):
        height = rect.get_height()
        tp_count = metrics_df['TP'].iloc[i]
        fn_count = metrics_df['FN'].iloc[i]
        total_count = metrics_df['Total'].iloc[i]
        
        # Text for True Predictions
        ax1.text(rect.get_x() + rect.get_width() / 2.0, 0.5, f'{tp_count}/{total_count}', ha='center', va='center', color='white', fontsize=10, rotation=90)

    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()

    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    if model_name is None:
        model_name = timestamp

    # Save the plot
    plt.savefig(f'results/metrics_plot_{model_name}.png')
    
    try:
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'font.size': 11,
            'text.usetex': True,
            'pgf.rcfonts': False,
        })
        
        plt.savefig(f'results/metrics_plot_{model_name}.pgf')
    except:
        pass


def generate_latex_table(metrics_df, model_name=None):
    # Sort by recall in descending order
    metrics_df = metrics_df.sort_values(by='Recall', ascending=False)

    # Calculate True Positives (TP) and False Negatives (FN) as a proportion of Total
    tp_proportion = metrics_df['TP'] / metrics_df['Total']
    fn_proportion = metrics_df['FN'] / metrics_df['Total']

    # Start LaTeX table
    latex_code = "\\begin{table}[h!]\n\\centering\n"
    latex_code += "\\begin{tabular}{|l|r|r|r|r|}\n"
    latex_code += "\\hline\n"
    latex_code += "Family & True Positives & False Negatives & TP Proportion & FN Proportion \\\\\n"
    latex_code += "\\hline\n"

    # Add data rows
    for i, row in metrics_df.iterrows():
        family = row['Family']
        tp_count = row['TP']
        fn_count = row['FN']
        total_count = row['Total']
        tp_prop = f"{tp_proportion[i]:.2f}"
        fn_prop = f"{fn_proportion[i]:.2f}"

        latex_code += f"{family} & {tp_count}/{total_count} & {fn_count}/{total_count} & {tp_prop} & {fn_prop} \\\\\n"

    latex_code += "\\hline\n"
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{True and False Predictions per Family}\n"
    latex_code += "\\label{tab:metrics}\n"
    latex_code += "\\end{table}\n"

    # Optionally save the LaTeX code to a file
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
    if model_name is None:
        model_name = timestamp

    with open(f'results/metrics_table_{model_name}.tex', 'w') as f:
        f.write(latex_code)

    print("LaTeX table generated and saved as a .tex file.")
    return latex_code