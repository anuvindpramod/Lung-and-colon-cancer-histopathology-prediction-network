# test.py

import torch
import os
import random
import numpy as np
from tqdm import tqdm
import config # Use updated config
from dataset import HistopathDataset, get_transforms, create_dataloader # Use cleaned dataset functions
from model import HistopathModel # Use model definition
# Import the metrics calculation function from train.py
from train import calculate_and_print_metrics

def evaluate_on_test_set(device_str=config.DEVICE):
    """
    Loads the best model and evaluates it on the held-out test set.
    """
    device = torch.device(device_str)
    print(f"--- Starting Test Set Evaluation ---")
    print(f"Using device: {device}")

    # --- Load Test Data ---
    print("\nLoading Test Dataset...")
    try:
        test_dataset = HistopathDataset(
            root_dir=config.DATASET_ROOT,
            transform=get_transforms('test'), # Use validation/test transforms
            split='test' # Specify the test split
        )
        if len(test_dataset) == 0:
            print("Error: Test dataset is empty. Check dataset splitting/paths.")
            return

        test_loader = create_dataloader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        print(f"Test dataset loaded with {len(test_dataset)} samples.")
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Initialize Model ---
    print("\nInitializing Model Architecture...")
    model = HistopathModel(num_classes=config.NUM_CLASSES).to(device)

    # --- Load Best Weights ---
    best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)
    print(f"\nLoading best saved weights from: {best_weights_path}")
    if os.path.exists(best_weights_path):
        try:
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
            print("Best weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Cannot perform test evaluation without loaded weights.")
            return
    else:
        print(f"Error: Best weights file not found at {best_weights_path}")
        print("Cannot perform test evaluation.")
        return

    # --- Evaluation Loop ---
    model.eval() # Set model to evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []
    print("\nRunning inference on test set...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing', leave=False, ncols=100)
        for images, labels in pbar:
            # Handle potential empty batches from collate_fn
            if images.numel() == 0:
                 continue

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # --- Calculate and Print Metrics ---
    print("\n--- Test Set Results ---")
    if all_labels and all_preds:
        # Use the existing function from train.py
        test_metrics = calculate_and_print_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            config.CLASSES
        )
        # You can optionally save these metrics to a file here
        # For example:
        # import json
        # with open("test_metrics_results.json", "w") as f:
        #     # Convert numpy arrays in confusion matrix to lists for JSON serialization
        #     if 'confusion_matrix' in test_metrics:
        #          test_metrics['confusion_matrix'] = test_metrics['confusion_matrix'].tolist()
        #     json.dump(test_metrics, f, indent=4)
        # print("\nTest metrics saved to test_metrics_results.json")

    else:
        print("No samples processed from the test set. Cannot calculate metrics.")

    print("\n--- Test Evaluation Complete ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Set seed for consistency if any randomness were involved (though eval should be deterministic)
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    evaluate_on_test_set()