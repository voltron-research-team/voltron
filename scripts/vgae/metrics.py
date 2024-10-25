import torch
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_accuracy(model, loader):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            logits = model.classify(z, data.batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def calculate_metrics(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)
            logits = model.classify(z, data.batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    accuarcy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    
    return accuarcy, recall, precision, f1, fpr
