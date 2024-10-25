from test import test_siamese_network
import numpy as np
import args
from sklearn.metrics import f1_score, precision_score, recall_score

def train_siamese_network(model, data_loader, optimizer, criterion, device, num_epochs=args.num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_labels = []
        all_predictions = []
        
        test_embeddings_path = args.test_embeddings_path
        support_set_path = args.support_set_path

        test_siamese_network(model, test_embeddings_path, support_set_path ,criterion, device)
        
        for embedding1, embedding2, label in data_loader:
            embedding1, embedding2, label = embedding1.to(device), embedding2.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(embedding1, embedding2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Collect predictions and labels for metric calculations
            predictions = (output > args.threshold).float()
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
        
        accuracy = (np.array(all_predictions) == np.array(all_labels)).mean()
        f1 = f1_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')