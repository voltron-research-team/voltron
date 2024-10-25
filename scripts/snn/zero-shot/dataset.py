from torch.utils.data import Dataset
import numpy as np
import torch
import args
import random
import os

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
    def __init__(self, embeddings, support_set, support_set_size=50, seed=args.seed):
        self.embeddings = embeddings
        self.support_set = support_set
        self.num_embeddings = len(embeddings)
        self.support_set_size = support_set_size

        if len(support_set) < support_set_size:
            raise ValueError("The support set size is greater than the number of elements in the support set")
        
        np.random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        return self.num_embeddings

    def __getitem__(self, idx):
        emb1 = self.embeddings[idx]
        
        # Select n random elements from the support set
        random_samples = np.random.choice(len(self.support_set), self.support_set_size, replace=False)
        sampled_embeddings = [self.support_set[i][0] for i in random_samples]

        if len(emb1) >= 3:
            path = emb1[2] 
        else:
            path = ''

        return emb1[0], sampled_embeddings, torch.tensor([emb1[1] != self.support_set[random_samples[0]][1]], dtype=torch.float32), path
    

class FewShotDataset(Dataset):
    def __init__(self, embeddings, support_set, shot_size, benign_shot_size=None, seed=args.seed, familyName=None):
        self.embeddings = embeddings
        self.embeddings2 = support_set  # benign support set
        self.embeddings3 = []  # malware support se
        self.num_embeddings = len(embeddings)
        self.shot_size = shot_size
        self.familyName = familyName.lower()
        
        # embedigns -> path -> familyName -> FamilyName ayni olanlari -> sameFamilyEmbeddings

        if benign_shot_size is None:
            self.benign_shot = shot_size
        else:
            self.benign_shot = benign_shot_size

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Discard n members with label 1 and add them to embeddings3
        self.discard_and_store()

        # Check for uniform shape in embeddings2 and embeddings3
        self.check_uniform_shape(self.embeddings2, 'embeddings2')
        self.check_uniform_shape(self.embeddings3, 'embeddings3')

        # Check if support sets have enough samples
        if len(self.embeddings2) < self.benign_shot:
            raise ValueError(f"Support set 'embeddings2' has fewer elements ({len(self.embeddings2)}) than the required benign_shot size ({self.benign_shot}).")
        if len(self.embeddings3) < self.shot_size:
            raise ValueError(f"Support set 'embeddings3' has fewer elements ({len(self.embeddings3)}) than the required shot_size ({self.shot_size}).")
        
        print(f"Initialized FewShotDataset with {len(self.embeddings)} test samples, {len(self.embeddings2)} benign support set samples, and {len(self.embeddings3)} malware support set samples.")

    def discard_and_store(self):
        # Shuffle embeddings using np.random
        np.random.shuffle(self.embeddings)
        for i in reversed(range(len(self.embeddings))):
            if self.embeddings[i][1] == torch.tensor(1, dtype=torch.float32) and len(self.embeddings3) < self.shot_size:
                embedding_path = self.embeddings[i][2]
        
                # Split the path to extract relevant components
                path_parts = embedding_path.split(os.sep)
                
                if 'malware_familied' in path_parts:
                    embedding_family_name = path_parts[-2]  # Family name is the parent folder of the file
                elif 'benignware' in path_parts:
                    embedding_family_name = None
                else:
                    raise ValueError(f"Invalid path structure for path: {embedding_path}")
                
                if embedding_family_name.lower() == self.familyName:
                    self.embeddings3.append(self.embeddings.pop(i))

        print(f"Discarded {len(self.embeddings3)} samples with label 1 and stored them in 'embeddings3'.")

    def check_uniform_shape(self, support_set, name):
        """Ensure all elements in the support set have the same shape."""
        if len(support_set) > 1:
            first_shape = support_set[0][0].shape
            for i, emb in enumerate(support_set):
                if emb[0].shape != first_shape:
                    raise ValueError(f"Inconsistent shape found in {name} at index {i}: Expected shape {first_shape}, but got {emb[0].shape}.")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        
        # Validate index
        if not isinstance(idx, int) or idx < 0 or idx >= len(self.embeddings):
            raise ValueError(f"Invalid index: {idx}. It must be a non-negative integer within the range of embeddings.")
        
        emb1 = self.embeddings[idx]

        # Validate emb1 shape
        if len(emb1) < 2:
            raise ValueError(f"emb1 at index {idx} is expected to have at least 2 elements, but got {len(emb1)}.")
        if not isinstance(emb1[0], torch.Tensor):
            raise ValueError(f"Expected emb1[0] to be a torch Tensor, but got {type(emb1[0])}.")
        
        # Validate the shot sizes
        if not isinstance(self.benign_shot, int) or self.benign_shot <= 0:
            raise ValueError(f"benign_shot must be a positive integer, but got {self.benign_shot}.")
        if not isinstance(self.shot_size, int) or self.shot_size <= 0:
            raise ValueError(f"shot_size must be a positive integer, but got {self.shot_size}.")

        # Ensure there are enough samples to choose from
        if len(self.embeddings2) < self.benign_shot:
            raise ValueError(f"Not enough benign samples: requested {self.benign_shot}, but only {len(self.embeddings2)} available.")
        if len(self.embeddings3) < self.shot_size:
            raise ValueError(f"Not enough malware samples: requested {self.shot_size}, but only {len(self.embeddings3)} available.")

        # Select random elements from the updated support set
        benign_samples = random.sample(self.embeddings2, self.benign_shot)
        malware_samples = random.sample(self.embeddings3, self.shot_size)

        # Validate that samples have the correct shape
        for i, sample in enumerate(benign_samples):
            if not isinstance(sample[0], torch.Tensor):
                raise ValueError(f"Expected benign sample {i} to be a torch Tensor, but got {type(sample[0])}.")
            if sample[0].shape != emb1[0].shape:
                raise ValueError(f"Shape mismatch in benign sample at index {i}: Expected shape {emb1[0].shape}, but got {sample[0].shape}.")
        
        for i, sample in enumerate(malware_samples):
            if not isinstance(sample[0], torch.Tensor):
                raise ValueError(f"Expected benign sample {i} to be a torch Tensor, but got {type(sample[0])}.")
            if sample[0].shape != emb1[0].shape:
                raise ValueError(f"Shape mismatch in malware sample at index {i}: Expected shape {emb1[0].shape}, but got {sample[0].shape}.")
        
        if len(emb1) >= 3:
            path = emb1[2] 
        else:
            path = ''

        # Debugging: Check for consistent shapes in benign and malware samples
        benign_shapes = [sample[0].shape for sample in benign_samples]
        malware_shapes = [sample[0].shape for sample in malware_samples]
        
        if len(set(benign_shapes)) > 1:
            raise ValueError(f"Inconsistent shapes found in benign samples: {benign_shapes}")
        if len(set(malware_shapes)) > 1:
            raise ValueError(f"Inconsistent shapes found in malware samples: {malware_shapes}")

        return emb1[0], [sample[0] for sample in benign_samples], [sample[0] for sample in malware_samples], emb1[1], path
