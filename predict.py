# predict.py
import torch
import os
from PIL import Image
from tqdm import tqdm
import config # Use updated config
from model import HistopathModel # Use cleaned model
from dataset import get_transforms # Use cleaned dataset transforms

def predict_batch(list_of_image_paths, model=None, device_str=config.DEVICE):
    """
    Predicts the class for a list of image file paths using the trained model.

    Loads the best validation model if available, otherwise the final model.

    Args:
        list_of_image_paths (list): List of strings, paths to image files.
                                    These should be paths to images like those
                                    in the submission 'data/' directory.
        model (torch.nn.Module, optional): Pre-loaded model. Defaults to None.
        device_str (str, optional): Device ('cuda', 'mps', 'cpu'). Defaults to config.DEVICE.

    Returns:
        list: A list of predicted class names (strings).
    """
    device = torch.device(device_str)
    class_names = config.CLASSES # Get class names from config in correct order

    # --- Load Model if not provided ---
    if model is None:
        print("Loading model for prediction...")
        model = HistopathModel(num_classes=config.NUM_CLASSES)
        best_weights_path = os.path.join(config.CHECKPOINT_DIR, config.BEST_VAL_WEIGHTS_FILE)
        final_weights_path = os.path.join(config.CHECKPOINT_DIR, config.FINAL_WEIGHTS_FILE)

        load_path = None
        if os.path.exists(best_weights_path):
             load_path = best_weights_path
             print(f"  Attempting to load best validation weights: {load_path}")
        elif os.path.exists(final_weights_path):
             load_path = final_weights_path
             print(f"  Best weights not found. Attempting to load final epoch weights: {load_path}")
        else:
             # Critical error if no weights found
             print(f"\nError: No model weights file found!")
             print(f"  Checked paths:")
             print(f"    - {best_weights_path}")
             print(f"    - {final_weights_path}")
             print("  Cannot perform prediction. Please ensure the model is trained.")
             # Return errors for all inputs
             return ["Error: Model weights not found"] * len(list_of_image_paths)

        try:
            model.load_state_dict(torch.load(load_path, map_location=device))
            print(f"  Model weights loaded successfully from {load_path}")
        except Exception as e:
            print(f"\nError loading model weights from {load_path}: {e}")
            import traceback
            traceback.print_exc()
            return [f"Error: Failed to load weights"] * len(list_of_image_paths)

    # --- Prepare Model and Transforms ---
    model = model.to(device)
    model.eval() # Set model to evaluation mode is crucial for prediction
    # Use validation transforms (no augmentation) for prediction
    transform = get_transforms('val')

    # --- Process Images ---
    predictions = []
    print(f"\nPredicting classes for {len(list_of_image_paths)} images...")
    with torch.no_grad(): # Disable gradient calculations for inference
        for img_path in tqdm(list_of_image_paths, desc="Predicting", ncols=100):
            try:
                # Load image
                img = Image.open(img_path).convert('RGB')
                # Apply transformations and add batch dimension
                img_tensor = transform(img).unsqueeze(0).to(device)

                # Get model output (logits)
                output = model(img_tensor)
                # Get predicted class index
                _, pred_idx = torch.max(output, 1)
                pred_idx_item = pred_idx.item() # Get index as Python int

                # Map index to class name
                if 0 <= pred_idx_item < len(class_names):
                    predicted_class_name = class_names[pred_idx_item]
                else:
                    print(f"Warning: Predicted index {pred_idx_item} out of range for class names.")
                    predicted_class_name = "Error: Index out of range"

                predictions.append(predicted_class_name)

            except FileNotFoundError:
                print(f"\nError: Image file not found at {img_path}")
                predictions.append("Error: File Not Found")
            except Exception as e:
                print(f"\nError processing image {img_path}: {e}")
                predictions.append("Error: Processing Failed")

    print("Prediction complete.")
    return predictions


# --- Example Usage Block (for testing predict.py directly) ---
if __name__ == '__main__':
    print("\n--- Running Prediction Example ---")
    # This block runs only when predict.py is executed directly.
    # It assumes the 'data/' directory exists and contains sample images
    # structured as per the PDF requirements (e.g., data/colon_n/image.jpeg) [cite: 26, 37, 38, 39, 40]

    submission_data_dir = "data" # Directory required by PDF for samples
    example_image_paths = []

    if os.path.exists(submission_data_dir) and os.path.isdir(submission_data_dir):
         print(f"Looking for sample images in '{submission_data_dir}/<class_name>/...'")
         for class_name in config.CLASSES:
              class_folder = os.path.join(submission_data_dir, class_name)
              if os.path.isdir(class_folder):
                   try:
                        # Find the first .jpeg file in the class directory
                        first_image = next(f for f in os.listdir(class_folder)
                                           if f.lower().endswith(('.jpeg', '.jpg', '.png')))
                        example_image_paths.append(os.path.join(class_folder, first_image))
                        print(f"  Found sample: {os.path.join(class_folder, first_image)}")
                        # Limit to a few examples if needed
                        # if len(example_image_paths) >= 5: break
                   except StopIteration:
                        # No image found in this specific class folder
                        print(f"  Warning: No image file found in {class_folder}")
              else:
                   print(f"  Warning: Class directory not found: {class_folder}")
         # Limit total number of examples
         # example_image_paths = example_image_paths[:10]
    else:
         print(f"\nWarning: Submission 'data/' directory ('{submission_data_dir}') not found or not a directory.")
         print("Cannot run prediction example. Please create 'data/' with sample images per class.")


    if example_image_paths:
        print(f"\nAttempting to predict on {len(example_image_paths)} found sample images...")
        predicted_labels = predict_batch(example_image_paths)

        print("\n--- Example Predictions ---")
        if predicted_labels:
             for img_path, label in zip(example_image_paths, predicted_labels):
                  print(f"  Image: {os.path.relpath(img_path)} -> Predicted Class: {label}")
        else:
             print("Prediction function returned no labels.")

    else:
        print("\nNo example images found in the 'data/' directory structure to run prediction test.")
        print(f"Please ensure '{submission_data_dir}' exists and contains subfolders named after classes")
        print(f"({config.CLASSES}) each containing sample images.")