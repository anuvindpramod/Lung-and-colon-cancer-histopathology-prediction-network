# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import os
import time
import random
import config # Use updated config
from dataset import HistopathDataset, get_transforms, create_dataloader # Use cleaned dataset
from model import HistopathModel # Use cleaned model

# --- Helper function for calculating and printing metrics ---
def calculate_and_print_metrics(y_true, y_pred, y_prob, class_names):
    """ Calculates and prints multi-class metrics including specificity """
    metrics = {}
    print("-" * 30)
    print("Validation Metrics:")
    # Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = accuracy
    print(f"  Overall Accuracy: {accuracy:.4f}")

    # Classification Report (Precision, Recall/Sensitivity, F1 per class, macro/weighted avg)
    # Ensure labels match the range of y_true/y_pred if using indices
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    target_names_filtered = [class_names[i] for i in unique_labels if i < len(class_names)]
    
    report = classification_report(y_true, y_pred, target_names=target_names_filtered, labels=unique_labels, zero_division=0, output_dict=True)
    print("\n  Classification Report:")
    for label, scores in report.items():
        if label in ['macro avg', 'weighted avg']:
            print(f"    {label.replace(' avg', '').capitalize()} Avg:")
            print(f"      Precision: {scores['precision']:.4f}")
            print(f"      Recall:    {scores['recall']:.4f}")
            print(f"      F1-Score:  {scores['f1-score']:.4f}")
        elif label != 'accuracy':
             # Check if the label corresponds to a known class name
             class_label_name = class_names[int(label)] if label.isdigit() and int(label) < len(class_names) else label
             print(f"    Class: {class_label_name} ({label})")
             print(f"      Precision: {scores['precision']:.4f}")
             print(f"      Recall (Sens): {scores['recall']:.4f}")
             print(f"      F1-Score:  {scores['f1-score']:.4f}")
             print(f"      Support:   {scores['support']}")


    metrics['report'] = report
    metrics['macro_f1'] = report['macro avg']['f1-score'] # Use macro F1 for saving best model

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels) # Use unique labels found
    print("\n  Confusion Matrix:")
    # Print CM header based on filtered names
    cm_header = " ".join([f"{name[:5]:>5}" for name in target_names_filtered])
    print(f"        Predicted ->\n         {cm_header}")
    for i, row in enumerate(cm):
        row_str = " ".join([f"{x:>5}" for x in row])
        print(f"True {target_names_filtered[i][:5]:<5} | {row_str}")

    metrics['confusion_matrix'] = cm

    # Specificity per class
    specificity_scores = {}
    print("\n  Specificity:")
    # Calculate using the full CM shape based on all classes, even if some not predicted/true
    num_all_classes = len(class_names)
    full_cm = confusion_matrix(y_true, y_pred, labels=range(num_all_classes))

    for i in range(num_all_classes):
        tn = full_cm.sum() - (full_cm[i, :].sum() + full_cm[:, i].sum() - full_cm[i, i])
        fp = full_cm[:, i].sum() - full_cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_scores[class_names[i]] = spec
        print(f"    Class '{class_names[i]}': {spec:.4f}")
    metrics['specificity'] = specificity_scores

    # AUC (One-vs-Rest Macro Average)
    try:
        # Ensure y_prob has scores for all classes defined in config
        if y_prob.shape[1] != num_all_classes:
             # Pad probabilities if some classes were never predicted (unlikely with softmax but safe)
             padded_probs = np.zeros((y_prob.shape[0], num_all_classes))
             unique_pred_classes = np.unique(y_pred) # Classes actually predicted
             # This part needs careful handling if classes are missing - simpler to just use OvR directly
             print("Warning: Probability shape mismatch for AUC. Skipping padding (might be complex).")


        auc_ovr_macro = roc_auc_score(
            np.eye(num_all_classes)[y_true], # One-hot encode true labels based on ALL classes
            y_prob, # Use the original probabilities
            multi_class='ovr',
            average='macro'
        )
        metrics['auc_ovr_macro'] = auc_ovr_macro
        print(f"\n  AUC (Macro OVR): {auc_ovr_macro:.4f}")
    except ValueError as e:
         # Often happens if only one class present in y_true during validation batch/epoch
         metrics['auc_ovr_macro'] = 0.0
         print(f"\n  AUC Calculation Warning: {e} (is only one class present?). Setting AUC to 0.0")

    print("-" * 30)
    return metrics


# --- Training and Validation Epoch Functions ---
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Runs a single training epoch."""
    model.train() # Set model to training mode
    running_loss = 0.0
    num_samples = 0

    pbar = tqdm(train_loader, desc='Training', leave=False, ncols=100)
    for images, labels in pbar:
        # Handle potential empty batches from collate_fn
        if images.numel() == 0:
             continue # Skip empty batch

        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_samples += batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return epoch_loss


def validate_epoch(model, val_loader, criterion, device, class_names):
    """Runs a single validation epoch and calculates metrics."""
    model.eval() # Set model to evaluation mode
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False, ncols=100)
        for images, labels in pbar:
            # Handle potential empty batches from collate_fn
            if images.numel() == 0:
                 continue

            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    print(f"\n  Validation Loss: {epoch_loss:.4f}")

    # Calculate detailed metrics if validation occurred
    metrics = {}
    if all_labels and all_preds:
        metrics = calculate_and_print_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            class_names
        )
        metrics['loss'] = epoch_loss # Add loss to the metrics dict
    else:
        print("  Validation resulted in no samples processed. Skipping metric calculation.")
        metrics['loss'] = epoch_loss
        metrics['macro_f1'] = 0.0 # Default value if no validation samples

    return metrics


# --- Main Training Function ---
def train_model(num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, device_str=config.DEVICE):
    """
    Main function to train the model.
    Handles dataset loading, model initialization, training loop, validation,
    LR scheduling, early stopping, and model saving.
    """
    # --- Setup ---
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(device_str)
    print(f"Using device: {device}")
    # Set seed for reproducibility across libraries
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    # --- Create Datasets and Dataloaders ---
    print("\n--- Loading Data ---")
    try:
        train_dataset = HistopathDataset(
            root_dir=config.DATASET_ROOT,
            transform=get_transforms('train'), 
            split='train'
        )
        val_dataset = HistopathDataset(
            root_dir=config.DATASET_ROOT,
            transform=get_transforms('val'),
            split='val'
        )
        # Ensure datasets loaded correctly
        if len(train_dataset) == 0 or len(val_dataset) == 0:
             raise ValueError("Training or validation dataset is empty after initialization.")

        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        print("Datasets and Dataloaders created successfully.")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"\nError loading data: {e}")
        print("Please check DATASET_ROOT in config.py and the dataset structure/contents.")
        import traceback
        traceback.print_exc()
        return None # Stop training if data loading fails

    print("\n--- Initializing Model ---")
    model = HistopathModel(num_classes=config.NUM_CLASSES).to(device)
    best_weights_path_to_load=os.path.join(config.CHECKPOINT_DIR,config.BEST_VAL_WEIGHTS_FILE)
    if os.path.exists(best_weights_path_to_load):
        try:
            model.load_state_dict(torch.load(best_weights_path_to_load, map_location=device))
            print("Successfully loaded pre-trained weights for fine-tuning.")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights from {best_weights_path_to_load}. Error: {e}")
    else:
        print(f"No pre-trained weights found at {best_weights_path_to_load}. Starting training from scratch.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Optimizer: Adam (LR={config.LEARNING_RATE}, WD={config.WEIGHT_DECAY})")
    print(f"  Scheduler: StepLR (Step={config.SCHEDULER_STEP}, Gamma={config.SCHEDULER_GAMMA})")
    print(f"  Loss Function: CrossEntropyLoss")

    # --- Training Loop ---
    best_val_metric = 0.0  # Track best Macro F1-score on validation set
    patience_counter = 0
    final_weights_path = os.path.join(config.CHECKPOINT_DIR, config.FINAL_WEIGHTS_FILE)
    best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)

    print("\n--- Starting Training Loop ---")
    start_training_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        # Train one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate one epoch
        val_metrics = validate_epoch(model, val_loader, criterion, device, config.CLASSES)

        # Step the LR scheduler
        scheduler.step()

        epoch_end_time = time.time()
        # --- Logging ---
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        # Validation metrics are printed within validate_epoch
        print(f"  Time: {epoch_end_time - epoch_start_time:.2f}s")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # --- Save Best Model based on Validation Macro F1 ---
        current_val_metric = val_metrics.get('macro_f1', 0.0)
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            try:
                torch.save(model.state_dict(), best_weights_path)
                print(f"  Saving new best model to {best_weights_path} (Macro F1: {best_val_metric:.4f})")
                patience_counter = 0 # Reset patience
            except Exception as e:
                print(f"  Error saving best model checkpoint: {e}")
        else:
            patience_counter += 1
            print(f"  Validation Macro F1 did not improve. Patience: {patience_counter}/{config.PATIENCE}")

        # --- Early Stopping Check ---
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs due to lack of improvement.")
            break

    total_training_time = time.time() - start_training_time
    print(f"\n--- Finished Training ---")
    print(f"Total training time: {total_training_time:.2f}s")

    # --- Save Final Model Weights ---
    # Always save the model state from the last completed epoch
    print(f"Saving final model weights from last epoch to {final_weights_path}...")
    try:
        torch.save(model.state_dict(), final_weights_path)
        print("  Final model weights saved successfully.")
    except Exception as e:
        print(f"  Error saving final model weights: {e}")

    # --- Load Best Model for Return ---
    # If a best model was saved based on validation, load those weights back
    if os.path.exists(best_weights_path):
        print(f"\nLoading best model weights from {best_weights_path} (Val Macro F1: {best_val_metric:.4f})...")
        try:
             # Load state dict requires model instance
             model.load_state_dict(torch.load(best_weights_path, map_location=device))
             print("  Best weights loaded successfully.")
             return model # Return the best model
        except Exception as e:
             print(f"  Warning: Could not load best weights from {best_weights_path}. Error: {e}")
             print("  Returning model from the last epoch instead.")
             # Ensure last epoch model is loaded if best fails (it should already be in 'model')
             return model
    else:
        print("\nNo best model checkpoint was saved (or initial metric was best).")
        print("Returning model from the last epoch.")
        return model # Return the model from the last epoch


# --- Script Entry Point ---
if __name__ == "__main__":
    print("="*50)
    print("Running Histopathology Model Training Script")
    print("="*50)
    # Ensure DATASET_ROOT is set correctly in config.py before running!
    if not os.path.exists(config.DATASET_ROOT):
         print(f"Error: DATASET_ROOT '{config.DATASET_ROOT}' not found!")
         print("Please update DATASET_ROOT in config.py")
    else:
        trained_model = train_model()
        if trained_model:
            print("\nTraining script completed.")
            # You could add a final evaluation on the test set here if desired,
            # but the user wanted to keep test set separate for now.
            # e.g., test_dataset = HistopathDataset(...)
            #      test_loader = create_dataloader(test_dataset, ...)
            #      test_metrics = validate_epoch(trained_model, test_loader, ...)
            #      print("\n--- Final Test Set Evaluation ---")
            #      # print test metrics
        else:
            print("\nTraining script failed.")