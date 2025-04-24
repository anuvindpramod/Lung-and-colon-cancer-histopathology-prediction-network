# dataset.py
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
    """Calculates the mean and std deviation of the training dataset."""
    print(f"Calculating dataset normalization statistics for images in: {dataset_root}")
    print(f"Using image size: {img_size}x{img_size}")


    basic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # Create a dataset instance loading ALL data first to ensure correct splitting
    # The perform_split=False prevents the dataset class from splitting internally here
    temp_dataset = HistopathDataset(root_dir=dataset_root, transform=basic_transform, split='all', perform_split=False)

    if len(temp_dataset.data) == 0:
        print("\nError: Cannot calculate stats, no images found.")
        print(f"Please check DATASET_ROOT in config.py ('{dataset_root}') and the directory structure.")
        return None, None

  
    all_indices = list(range(len(temp_dataset)))
    all_labels = [label for _, label in temp_dataset.data]

 
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_SEED,
        stratify=all_labels
    )

    # Extract labels corresponding to the train_val indices for the second split
    train_val_labels = [all_labels[i] for i in train_val_indices]

    # Second split: Train vs Val indices (from Train+Val set)
    # Calculate proportion for validation split from the train_val set
    val_proportion_of_intermediate = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_proportion_of_intermediate,
        random_state=config.RANDOM_SEED,
        stratify=train_val_labels
    )
    # --- End of splitting ---

    print(f"\nSplitting results for calculation:")
    print(f"  Total images: {len(all_indices)}")
    print(f"  Train images: {len(train_indices)}")
    print(f"  Validation images: {len(val_indices)}")
    print(f"  Test images: {len(test_indices)}")
    print(f"\nCalculating stats based on {len(train_indices)} training images...")

    # Create a Subset using only the training indices
    train_subset_for_stats = Subset(temp_dataset, train_indices)

    # Use a DataLoader to iterate efficiently
    # Set num_workers=0 to avoid potential issues during calculation
    stats_loader = DataLoader(train_subset_for_stats, batch_size=batch_size, shuffle=False, num_workers=0)

    # Variables to store sum and sum of squares per channel
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_pixels_total = 0
    processed_images = 0

    for images, _ in tqdm(stats_loader, desc="Calculating Stats"):
        # Skip potentially problematic batches (though shouldn't happen with Subset)
        if images is None or images.nelement() == 0:
            continue
        # Ensure images are float32 for calculations
        images = images.float()
        # images shape: [batch, channels, height, width]
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        channel_sum_sq += torch.sum(images ** 2, dim=[0, 2, 3])
        # Accumulate total number of pixels (per channel)
        num_pixels_total += images.shape[0] * images.shape[2] * images.shape[3]
        processed_images += images.shape[0]

    if num_pixels_total == 0 or processed_images == 0:
        print("\nError: No pixels processed during statistics calculation. Check DataLoader or Dataset.")
        return None, None

    # Calculate mean and std dev
    mean = channel_sum / num_pixels_total
    # Var = E[X^2] - (E[X])^2
    variance = (channel_sum_sq / num_pixels_total) - (mean ** 2)
    # Clamp variance to avoid negative values due to numerical instability
    std_dev = torch.sqrt(torch.clamp(variance, min=1e-6))

    mean_list = mean.tolist()
    std_list = std_dev.tolist()

    print(f"\nCalculation complete ({processed_images} images processed).")
    print(f"  Calculated Mean: {mean_list}")
    print(f"  Calculated Std Dev: {std_list}")
    print("\n--> IMPORTANT: Update DATASET_MEAN and DATASET_STD in config.py with these values! <---")

    return mean_list, std_list


# --- Transformations ---
def get_transforms(split):
    """Get appropriate transforms using dataset-specific mean/std from config."""
    # Using calculated mean/std now
    normalize_transform = transforms.Normalize(mean=config.DATASET_MEAN, std=config.DATASET_STD)

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5), # Enable horizontal flips
            transforms.RandomVerticalFlip(p=0.5),   # Enable vertical flips
            transforms.RandomRotation(degrees=20),  # Enable rotations (e.g., +/- 20 degrees)
            # Enable color jitter to simulate stain variations
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # ------------------------------------
            transforms.ToTensor(),
            normalize_transform # Apply normalization LAST
        ])
    else:  # For validation and test (NO augmentation here)
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            normalize_transform
        ])

# --- Dataset Class ---
class HistopathDataset(Dataset):
    """Loads histopathology images, handles class mapping, and enables splitting."""
    def __init__(self, root_dir, transform=None, split='train', perform_split=True):
        """
        Args:
            root_dir (str): Path to the dataset root (containing lung/colon sets).
            transform (callable, optional): Transformations to apply.
            split (str): 'train', 'val', 'test', or 'all'.
            perform_split (bool): If True, perform split based on 'split' arg.
                                  If False, load all data (used for stats calc).
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.class_to_idx = {} # Will be populated during loading
        self.idx_to_class = {} # Reverse mapping
        self.data = [] # List of tuples: (image_path, label_index)

        # --- Load image paths and assign labels ---
        self._load_data()

        if len(self.data) == 0:
             print(f"\nError: No image data loaded from {self.root_dir}. Aborting dataset initialization.")
             return # Stop if no data

        # --- Perform train/val/test split if requested ---
        if perform_split and self.split != 'all':
            self._split_data()
        elif self.split == 'all':
             print(f"Initialized dataset with all {len(self.data)} images (split='all').")


    def _load_data(self):
        """ Scans subdirectories and populates self.data and class mappings. """
        print(f"\nLoading image list from: {self.root_dir}")
        # Define class processing order and directories
        # This structure matches the user's description
        class_sources = [
            (os.path.join(self.root_dir, 'lung_image_sets'), ['lung_aca', 'lung_n', 'lung_scc']),
            (os.path.join(self.root_dir, 'colon_image_sets'), ['colon_aca', 'colon_n'])
        ]
        
        # Verify config classes match potential loaded classes
        expected_classes = set(config.CLASSES)
        found_classes = set()
        for _, classes_in_dir in class_sources:
            found_classes.update(classes_in_dir)
        
        if expected_classes != found_classes:
            print("\nWarning: Mismatch between classes in config.py and expected directory structure!")
            print(f"  Config expects: {sorted(list(expected_classes))}")
            print(f"  Code looks for: {sorted(list(found_classes))}")
            print("  Proceeding, but check config.CLASSES and your folder names.\n")


        current_idx = 0
        temp_class_to_idx = {} # Use temporary dict to build mapping

        for base_dir, classes_in_dir in class_sources:
            if not os.path.isdir(base_dir):
                print(f"  Warning: Base directory not found: {base_dir}, skipping...")
                continue

            for cls_name in classes_in_dir:
                # Only add class if it's expected by config.py
                if cls_name in expected_classes:
                     class_dir = os.path.join(base_dir, cls_name)
                     if os.path.isdir(class_dir):
                         # Assign index if class is new *and* in config.CLASSES
                         if cls_name not in temp_class_to_idx:
                             temp_class_to_idx[cls_name] = config.CLASSES.index(cls_name) # Use index from config order
                             print(f"  Mapping class '{cls_name}' to index {temp_class_to_idx[cls_name]}")

                         cls_idx = temp_class_to_idx[cls_name]
                         # Load image paths for this class
                         img_count = 0
                         for img_name in os.listdir(class_dir):
                             if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                                 self.data.append((os.path.join(class_dir, img_name), cls_idx))
                                 img_count += 1
                         print(f"    Found {img_count} images in {cls_name}")
                     else:
                          print(f"  Warning: Class directory not found: {class_dir}")
                else:
                    print(f"  Skipping directory '{cls_name}' as it's not in config.CLASSES.")

        # Finalize class mappings based on config order
        self.class_to_idx = {name: i for i, name in enumerate(config.CLASSES)}
        self.idx_to_class = {i: name for i, name in enumerate(config.CLASSES)}


    def _split_data(self):
        """ Performs stratified train/val/test split on self.data """
        print(f"\nPerforming data split for '{self.split}'...")
        if len(self.data) < 2: # Cannot split if less than 2 samples
            print("Warning: Not enough data to perform split.")
            return

        # Set seed for reproducible splits
        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)

        all_labels_for_split = [label for _, label in self.data]

        try:
            # First split: Train+Val vs Test
            train_val_data, test_data = train_test_split(
                self.data,
                test_size=config.TEST_SPLIT,
                random_state=config.RANDOM_SEED,
                stratify=all_labels_for_split
            )

            # Handle edge case where train_val might be too small for next split
            if len(train_val_data) < 2:
                 print("Warning: train_val split is too small for further val split.")
                 # Assign based on requested split, potentially leaving one empty
                 if self.split == 'train': self.data = train_val_data
                 elif self.split == 'test': self.data = test_data
                 else: self.data = [] # Validation set becomes empty
                 print(f"Assigned {len(self.data)} images to '{self.split}' split after initial split.")
                 return

            # Second split: Train vs Val (from Train+Val set)
            train_val_labels = [label for _, label in train_val_data]
            # Calculate proportion for validation split from the train_val set
            val_proportion_of_intermediate = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)

            # Ensure proportion is valid
            if not (0 < val_proportion_of_intermediate < 1):
                 print(f"Warning: Invalid validation proportion ({val_proportion_of_intermediate:.2f}) derived from VAL_SPLIT/TEST_SPLIT. Adjusting.")
                 # Fallback: Maybe split 50/50 if possible, or assign all to train
                 val_proportion_of_intermediate = 0.5 if len(train_val_data) >= 2 else 0.0


            train_data, val_data = train_test_split(
                train_val_data,
                test_size=val_proportion_of_intermediate,
                random_state=config.RANDOM_SEED,
                stratify=train_val_labels
            )
        except ValueError as e:
            print(f"\nError during train/test split (possibly too few samples per class): {e}")
            print("Dataset splitting failed. Check dataset size and class distribution.")
            self.data = [] # Clear data to indicate failure
            return

        # Assign the correct data split
        if self.split == 'train':
            self.data = train_data
        elif self.split == 'val':
            self.data = val_data
        elif self.split == 'test':
            self.data = test_data
        else:
             print(f"Warning: Unknown split '{self.split}' requested during split assignment.")
             self.data = [] # Should not happen if called correctly

        print(f"Split complete. Assigned {len(self.data)} images to '{self.split}' split.")
        # Shuffle the selected split's data
        random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.data or idx >= len(self.data):
             raise IndexError("Dataset index out of range or dataset is empty.")

        img_path, label_idx = self.data[idx]
        try:
            # Open image and ensure it's RGB
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
             print(f"Error: Image file not found at {img_path}. Returning None.")
             return None, None
        except Exception as e:
             print(f"Warning: Error loading image {img_path}: {e}. Returning None.")
             return None, None # Return None pair to be handled by collate_fn

        # Apply transformations if they exist
        if self.transform:
            image = self.transform(image)

        # Return image and label as a LongTensor
        return image, torch.tensor(label_idx, dtype=torch.long)


# --- Custom Collate Function ---
def collate_fn(batch):
    """ Filters out None samples from a batch (resulting from image loading errors). """
    # Filter out samples where image or label loading failed (represented as None in the tuple)
    batch = [(img, lbl) for img, lbl in batch if img is not None and lbl is not None]

    if not batch:
        # Return empty tensors if the whole batch failed
        # Ensure tensors have correct number of dimensions even when empty
        return torch.empty(0, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE), torch.empty(0, dtype=torch.long)

    # Use default collate for the filtered list of valid samples
    return torch.utils.data.dataloader.default_collate(batch)

# --- DataLoader Creation Function ---
def create_dataloader(dataset, batch_size, shuffle=True):
    """Creates a DataLoader with the custom collate function."""
    if len(dataset) == 0:
        print(f"Warning: Attempting to create DataLoader for an empty dataset (split: {dataset.split}).")
        # Return an empty loader structure or handle appropriately
        # For simplicity, we'll let it proceed, but the training loop should handle empty loaders.
        pass

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE == 'cuda' else False, # Pin memory usually benefits CUDA
        collate_fn=collate_fn # Use the custom collate function
    )

# --- Example Usage Block ---
if __name__ == "__main__":
    # This block runs only when dataset.py is executed directly
    print("Running dataset.py directly...")

    # 1. Calculate Normalization Stats
    print("\n--- Attempting to Calculate Normalization Stats ---")
    if os.path.exists(config.DATASET_ROOT):
         calculate_norm_stats()
    else:
         print(f"\nError: Dataset root directory '{config.DATASET_ROOT}' not found.")
         print("Cannot calculate normalization stats.")
         print("Please set DATASET_ROOT correctly in config.py and ensure the directory exists.")

    # 2. Test Dataset Loading and Splitting
    print("\n--- Testing Dataset Loading and Splitting ---")
    try:
        print("\nLoading TRAIN split:")
        train_dataset = HistopathDataset(root_dir=config.DATASET_ROOT, transform=get_transforms('train'), split='train')
        if len(train_dataset) > 0:
            print(f"  Train dataset loaded successfully with {len(train_dataset)} samples.")
            train_loader = create_dataloader(train_dataset, batch_size=config.BATCH_SIZE)
            print("  Train DataLoader created.")
            # Try getting one batch
            try:
                img_batch, lbl_batch = next(iter(train_loader))
                print(f"  Successfully fetched one training batch: Images shape {img_batch.shape}, Labels shape {lbl_batch.shape}")
            except StopIteration:
                print("  Could not fetch a batch from training loader (dataset might be smaller than batch size).")
            except Exception as e:
                print(f"  Error fetching batch from training loader: {e}")

        print("\nLoading VAL split:")
        val_dataset = HistopathDataset(root_dir=config.DATASET_ROOT, transform=get_transforms('val'), split='val')
        if len(val_dataset) > 0:
            print(f"  Validation dataset loaded successfully with {len(val_dataset)} samples.")

        print("\nLoading TEST split:")
        test_dataset = HistopathDataset(root_dir=config.DATASET_ROOT, transform=get_transforms('test'), split='test')
        if len(test_dataset) > 0:
            print(f"  Test dataset loaded successfully with {len(test_dataset)} samples.")

        # Verify class mapping
        if hasattr(train_dataset, 'class_to_idx'):
            print("\nClass to Index mapping used:")
            print(train_dataset.class_to_idx)
            print("Check if this matches config.CLASSES order.")

    except Exception as e:
        print(f"\nAn error occurred during dataset loading/splitting test: {e}")
        import traceback
        traceback.print_exc()