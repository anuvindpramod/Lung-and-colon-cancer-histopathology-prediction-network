import torch
import os
import random
import numpy as np
from tqdm import tqdm
import config
from dataset import HistopathDataset, get_transforms, create_dataloader
from model import HistopathModel
from train import calculate_and_print_metrics

def evaluate_on_test_set(device_str=config.DEVICE):
    """Loads the best model and evaluates it on the held-out test set """
    device = torch.device(device_str)
    
    try:
        test_dataset = HistopathDataset(
            root_dir=config.DATASET_ROOT,
            transform=get_transforms('test'),
            split='test'
        )
        if len(test_dataset) == 0:
            print("Error: Empty test dataset")
            return

        test_loader = create_dataloader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    model = HistopathModel(num_classes=config.NUM_CLASSES).to(device)
    best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)
    
    if os.path.exists(best_weights_path):
        try:
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Error: Weights not found at {best_weights_path}")
        return

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', ncols=100):
            if images.numel() == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    if all_labels and all_preds:
        calculate_and_print_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            config.CLASSES
        )

# --- Script Entry Point 
if __name__ == "__main__":
    # Set seed for consistency if any randomness were involved (though eval should be deterministic)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    evaluate_on_test_set()