from torch.utils.data import Dataset
import random
import torch
import args
import numpy as np

class SiameseDataset(Dataset):
    def __init__(self, embeddings, seed=args.seed):
        self.embeddings = embeddings
        self.num_embeddings = len(embeddings)
        self.labels = np.array([emb[1] for emb in embeddings])
        self.class_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}
        self.different_class_indices = {label: np.where(self.labels != label)[0] for label in np.unique(self.labels)}
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.num_embeddings

    def __getitem__(self, idx):
        is_same_class = np.random.rand() > 0.5
        emb1 = self.embeddings[idx]
        emb1_label = int(emb1[1].item())  # Convert the label tensor to an integer

        if is_same_class:
            same_class_idx = np.random.choice(self.class_indices[emb1_label])
            emb2 = self.embeddings[same_class_idx]
        else:
            different_class_idx = np.random.choice(self.different_class_indices[emb1_label])
            emb2 = self.embeddings[different_class_idx]

        return emb1[0], emb2[0], torch.tensor([1 if is_same_class else 0], dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, embeddings,support_set):
        self.embeddings = embeddings
        self.embeddings2 = support_set # benings
        self.num_embeddings = len(embeddings)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    def __len__(self):
        return self.num_embeddings
        
    def __getitem__(self, idx):
        emb1 = self.embeddings[idx]
        
        # Select n random elements from the support set
        random_samples = random.sample(self.embeddings2, 10)

        return emb1[0], [sample[0] for sample in random_samples], torch.tensor([emb1[1] != random_samples[0][1]], dtype=torch.float32)
    
    
class FewShotDataset(Dataset):
    def __init__(self, embeddings, support_set, shot=0,benign_shot=10):
        self.embeddings = embeddings
        self.embeddings2 = support_set  # benign support set
        self.embeddings3 = []  # malware support set
        self.num_embeddings = len(embeddings)
        self.shot = shot
        self.benign_shot = benign_shot
        
        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Discard n members with label 1 and add them to embeddings3
        self.discard_and_store()

    def discard_and_store(self):
        # Iterate in reverse to safely remove items from the list while iterating
        random.shuffle(self.embeddings)
        for i in reversed(range(len(self.embeddings))):
            if self.embeddings[i][1] == torch.tensor(1, dtype=torch.float32) and len(self.embeddings3) < self.shot:
                self.embeddings3.append(self.embeddings.pop(i))

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb1 = self.embeddings[idx]

        # Select random elements from the updated support set
        benign_samples = random.sample(self.embeddings2, self.benign_shot)
        malware_samples = random.sample(self.embeddings3, self.shot)

        return emb1[0], [sample[0] for sample in benign_samples], [sample[0] for sample in malware_samples],emb1[1]