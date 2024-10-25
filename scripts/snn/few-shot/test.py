import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FewShotDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import args

def test_siamese_network(model, test_embeddings_path,support_set_path, criterion, device):
    # Load the test embeddings
    test_embeddings = torch.load(test_embeddings_path, map_location=torch.device('cpu'))
    support_set = torch.load(support_set_path, map_location=torch.device('cpu'))
   
    # Create the test dataset and loader
    test_dataset = FewShotDataset(test_embeddings,support_set)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)

    # Evaluate the model
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for embedding1, embedding2, embedding3,label in test_loader:
            #embedding1, embedding2, label = embedding1.to(device), embedding2.to(device), label.to(device)
                        # Move data to device if needed
            # embedding1, embedding2, embedding3, label = embedding1.to(device), embedding2.to(device), embedding3.to(device), label.to(device)

            # Compute outputs
            benign_output = sum([model(embedding1, embed) for embed in embedding2]) / len(embedding2)
            malware_output = sum([model(embedding1, embed) for embed in embedding3]) / len(embedding3)

            predictions = (malware_output >= benign_output).float()

            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions)
    
    average_loss = total_loss / len(test_loader)
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    false_positives = np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
    
    # Calculate false positive rate (FPR)
    FP = sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))
    TN = sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0))
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}, '
            f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, FPR: {FPR[0]*100:.4f}%, False Positives: {false_positives}')
    

    return average_loss, accuracy, precision, recall, f1
