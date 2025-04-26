import torch
import os
from PIL import Image
from tqdm import tqdm
import config 
from model import HistopathModel
from dataset import get_transforms 

def predict_batch(list_of_image_paths, model=None, device_str=config.DEVICE):
    device = torch.device(device_str)
    class_names = config.CLASSES 

    if model is None:
        model = HistopathModel(num_classes=config.NUM_CLASSES)
        best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)
        final_weights_path = os.path.join(config.CHECKPOINT_DIR, config.FINAL_WEIGHTS_FILE)

        load_path = None
        if os.path.exists(final_weights_path):
            load_path = final_weights_path
        elif os.path.exists(best_weights_path):
            load_path = best_weights_path
        else:
            print(f"Error: No model weights found in {config.CHECKPOINT_DIR}")
            return ["Error: Model weights not found"] * len(list_of_image_paths)

        try:
            model.load_state_dict(torch.load(load_path, map_location=device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            return [f"Error: Failed to load weights"] * len(list_of_image_paths)

    model = model.to(device)
    model.eval() 
    transform = get_transforms('val')

    predictions = []
    with torch.no_grad():
        for img_path in tqdm(list_of_image_paths, desc="Predicting", ncols=100):
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                output = model(img_tensor)
                _, pred_idx = torch.max(output, 1)
                pred_idx_item = pred_idx.item()
                
                if 0 <= pred_idx_item < len(class_names):
                    predicted_class_name = class_names[pred_idx_item]
                else:
                    predicted_class_name = "Error: Invalid prediction"
                predictions.append(predicted_class_name)
            except FileNotFoundError:
                predictions.append("Error: File Not Found")
            except Exception as e:
                predictions.append("Error: Processing Failed")

    return predictions

if __name__ == '__main__':
    submission_data_dir = "data"
    example_image_paths = []

    if os.path.exists(submission_data_dir) and os.path.isdir(submission_data_dir):
        for class_name in config.CLASSES:
            class_folder = os.path.join(submission_data_dir, class_name)
            if os.path.isdir(class_folder):
                try:
                    first_image = next(f for f in os.listdir(class_folder)
                                     if f.lower().endswith(('.jpeg', '.jpg', '.png')))
                    example_image_paths.append(os.path.join(class_folder, first_image))
                except StopIteration:
                    continue
            else:
                print(f"Warning: Directory not found: {class_folder}")
    else:
        print(f"Error: Data directory '{submission_data_dir}' not found")

    if example_image_paths:
        predicted_labels = predict_batch(example_image_paths)
        if predicted_labels:
            for img_path, label in zip(example_image_paths, predicted_labels):
                print(f"{os.path.relpath(img_path)} -> {label}")
    else:
        print(f"No images found in {submission_data_dir}")
