import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import config 

def calculate_norm_stats(dataset_root=config.DATASET_ROOT, img_size=config.IMG_SIZE, batch_size=32):
    basic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    temp_dataset = HistopathDataset(root_dir=dataset_root, transform=basic_transform, split='all', perform_split=False)

    if len(temp_dataset.data) == 0:
        print("Error: No images found")
        return None, None

    all_indices = list(range(len(temp_dataset)))
    all_labels = [label for _, label in temp_dataset.data]

    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=all_labels
    )

    train_val_labels = [all_labels[i] for i in train_val_indices]
    val_proportion_of_intermediate = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_proportion_of_intermediate,
        random_state=config.RANDOM_SEED,
        stratify=train_val_labels
    )

    train_subset_for_stats = Subset(temp_dataset, train_indices)
    stats_loader = DataLoader(train_subset_for_stats, batch_size=batch_size, shuffle=False, num_workers=0)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels_total = 0
    processed_images = 0

    for images, _ in tqdm(stats_loader, desc="Calculating Stats"):
        if images is None or images.nelement() == 0:
            continue
        images = images.float()
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images ** 2, dim=[0, 2, 3])
        num_pixels_total += images.shape[0] * images.shape[2] * images.shape[3]
        processed_images += images.shape[0]

    if num_pixels_total == 0:
        return None, None

    mean = channel_sum / num_pixels_total
    variance = (channel_sum_sq / num_pixels_total) - (mean ** 2)
    std_dev = torch.sqrt(torch.clamp(variance, min=1e-6))

    return mean.tolist(), std_dev.tolist()

def get_transforms(split):
    normalize_transform = transforms.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD)

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5), 
            transforms.RandomVerticalFlip(p=0.5),   
            transforms.RandomRotation(degrees=20), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize_transform 
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            normalize_transform
        ])

class HistopathDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', perform_split=True):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.data = []

        self._load_data()

        if len(self.data) == 0:
            return

        if perform_split and self.split != 'all':
            self._split_data()

    def _load_data(self):
        class_sources = [
            (os.path.join(self.root_dir, 'lung_image_sets'), ['lung_aca', 'lung_n', 'lung_scc']),
            (os.path.join(self.root_dir, 'colon_image_sets'), ['colon_aca', 'colon_n'])
        ]
        
        expected_classes = set(config.CLASSES)
        found_classes = set()
        for _, classes_in_dir in class_sources:
            found_classes.update(classes_in_dir)
        
        if expected_classes != found_classes:
            print(f"Warning: Class mismatch. Expected: {sorted(list(expected_classes))}, Found: {sorted(list(found_classes))}")

        temp_class_to_idx = {}

        for base_dir, classes_in_dir in class_sources:
            if not os.path.isdir(base_dir):
                continue

            for cls_name in classes_in_dir:
                if cls_name in expected_classes:
                    class_dir = os.path.join(base_dir, cls_name)
                    if os.path.isdir(class_dir):
                        if cls_name not in temp_class_to_idx:
                            temp_class_to_idx[cls_name] = config.CLASSES.index(cls_name)

                        cls_idx = temp_class_to_idx[cls_name]
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                                self.data.append((os.path.join(class_dir, img_name), cls_idx))

        self.class_to_idx = {name: i for i, name in enumerate(config.CLASSES)}
        self.idx_to_class = {i: name for i, name in enumerate(config.CLASSES)}

    def _split_data(self):
        if len(self.data) < 2:
            return

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        all_labels_for_split = [label for _, label in self.data]

        try:
            train_val_data, test_data = train_test_split(
                self.data,
                test_size=config.TEST_SPLIT,
                random_state=config.RANDOM_SEED,
                stratify=all_labels_for_split
            )
            
            if len(train_val_data) < 2:
                if self.split == 'train': self.data = train_val_data
                elif self.split == 'test': self.data = test_data
                else: self.data = []
                return

            train_val_labels = [label for _, label in train_val_data]
            val_proportion_of_intermediate = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)

            if not (0 < val_proportion_of_intermediate < 1):
                val_proportion_of_intermediate = 0.5 if len(train_val_data) >= 2 else 0.0

            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_proportion_of_intermediate,
                random_state=config.RANDOM_SEED,
                stratify=train_val_labels
            )
        except ValueError as e:
            self.data = []
            return

        if self.split == 'train':
            self.data = train_data
        elif self.split == 'val':
            self.data = val_data
        elif self.split == 'test':
            self.data = test_data
        else:
            self.data = []

        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.data or idx >= len(self.data):
            raise IndexError("Dataset index out of range or dataset is empty")

        img_path, label_idx = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return None, None
        except Exception:
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label_idx, dtype=torch.long)

def collate_fn(batch):
    batch = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]

    if not batch:
        return torch.empty(0, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE), torch.empty(0, dtype=torch.long)

    return torch.utils.data.dataloader.default_collate(batch)

def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=collate_fn
    )

if __name__ == "__main__":
    if os.path.exists(config.DATASET_ROOT):
        calculate_norm_stats()
    else:
        print(f"Error: Dataset directory not found at {config.DATASET_ROOT}")