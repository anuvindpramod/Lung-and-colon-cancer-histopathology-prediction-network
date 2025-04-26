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
import config
from dataset import HistopathDataset, get_transforms, create_dataloader 
from model import HistopathModel 

def calculate_and_print_metrics(y_true, y_pred, y_prob, class_names):
    metrics = {}
    print("-" * 30)
    print("Validation Metrics:")
    accuracy = accuracy_score(y_true, y_pred)
    metrics['accuracy'] = accuracy
    print(f"  Overall Accuracy: {accuracy:.4f}")
    
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
            class_label_name = class_names[int(label)] if label.isdigit() and int(label) < len(class_names) else label
            print(f"    Class: {class_label_name} ({label})")
            print(f"      Precision: {scores['precision']:.4f}")
            print(f"      Recall (Sens): {scores['recall']:.4f}")
            print(f"      F1-Score:  {scores['f1-score']:.4f}")
            print(f"      Support:   {scores['support']}")

    metrics['report'] = report
    metrics['macro_f1'] = report['macro avg']['f1-score']

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    print("\n  Confusion Matrix:")
    cm_header = " ".join([f"{name[:5]:>5}" for name in target_names_filtered])
    print(f"        Predicted ->\n         {cm_header}")
    for i, row in enumerate(cm):
        row_str = " ".join([f"{x:>5}" for x in row])
        print(f"True {target_names_filtered[i][:5]:<5} | {row_str}")

    metrics['confusion_matrix'] = cm

    specificity_scores = {}
    print("\n  Specificity:")
    num_all_classes = len(class_names)
    full_cm = confusion_matrix(y_true, y_pred, labels=range(num_all_classes))

    for i in range(num_all_classes):
        tn = full_cm.sum() - (full_cm[i, :].sum() + full_cm[:, i].sum() - full_cm[i, i])
        fp = full_cm[:, i].sum() - full_cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_scores[class_names[i]] = spec
        print(f"    Class '{class_names[i]}': {spec:.4f}")
    metrics['specificity'] = specificity_scores

    try:
        if y_prob.shape[1] != num_all_classes:
            print("Warning: Probability shape mismatch for AUC. Skipping padding.")

        auc_ovr_macro = roc_auc_score(
            np.eye(num_all_classes)[y_true],
            y_prob,
            multi_class='ovr',
            average='macro'
        )
        metrics['auc_ovr_macro'] = auc_ovr_macro
        print(f"\n  AUC (Macro OVR): {auc_ovr_macro:.4f}")
    except ValueError as e:
        metrics['auc_ovr_macro'] = 0.0
        print(f"\n  AUC Calculation Warning: {e}")

    print("-" * 30)
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_samples = 0

    pbar = tqdm(train_loader, desc='Training', leave=False, ncols=100)
    for images, labels in pbar:
        if images.numel() == 0:
            continue

        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        num_samples += batch_size
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    return epoch_loss

def validate_epoch(model, val_loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_samples = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False, ncols=100)
        for images, labels in pbar:
            if images.numel() == 0:
                continue

            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / num_samples if num_samples > 0 else 0.0
    print(f"\n  Validation Loss: {epoch_loss:.4f}")

    metrics = {}
    if all_labels and all_preds:
        metrics = calculate_and_print_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            class_names
        )
        metrics['loss'] = epoch_loss
    else:
        print("  No samples processed during validation.")
        metrics['loss'] = epoch_loss
        metrics['macro_f1'] = 0.0

    return metrics

def train_model(num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE, device_str=config.DEVICE):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(device_str)
    print(f"Using device: {device}")

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

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

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("Training or validation dataset is empty.")

        train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")

    except Exception as e:
        print(f"\nError loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

    print("\n--- Initializing Model ---")
    model = HistopathModel(num_classes=config.NUM_CLASSES).to(device)
    best_weights_path_to_load = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)
    
    if os.path.exists(best_weights_path_to_load):
        try:
            model.load_state_dict(torch.load(best_weights_path_to_load, map_location=device))
            print("Successfully loaded pre-trained weights.")
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
    else:
        print("Starting training from scratch.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=config.SCHEDULER_STEP, gamma=config.SCHEDULER_GAMMA)

    best_val_metric = 0.0 
    patience_counter = 0
    final_weights_path = os.path.join(config.CHECKPOINT_DIR, config.FINAL_WEIGHTS_FILE)
    best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)

    print("\n--- Starting Training Loop ---")
    start_training_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device, config.CLASSES)
        scheduler.step()

        epoch_end_time = time.time()
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Time: {epoch_end_time - epoch_start_time:.2f}s")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        current_val_metric = val_metrics.get('macro_f1', 0.0)
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            try:
                torch.save(model.state_dict(), best_weights_path)
                print(f"  Saving new best model (Macro F1: {best_val_metric:.4f})")
                patience_counter = 0
            except Exception as e:
                print(f"  Error saving best model: {e}")
        else:
            patience_counter += 1
            print(f"  Validation Macro F1 did not improve. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    total_training_time = time.time() - start_training_time
    print(f"\n--- Finished Training ---")
    print(f"Total training time: {total_training_time:.2f}s")

    try:
        torch.save(model.state_dict(), final_weights_path)
        print("Final model weights saved.")
    except Exception as e:
        print(f"Error saving final weights: {e}")

    if os.path.exists(best_weights_path):
        try:
            model.load_state_dict(torch.load(best_weights_path, map_location=device))
            print("Best weights loaded.")
            return model
        except Exception as e:
            print(f"Warning: Could not load best weights: {e}")
            return model
    else:
        return model

if __name__ == "__main__":
    if not os.path.exists(config.DATASET_ROOT):
        print(f"Error: DATASET_ROOT '{config.DATASET_ROOT}'")
    else:
        trained_model = train_model()
        if trained_model:
            print("\nTraining completed successfully.")
        else:
            print("\nTraining failed.")