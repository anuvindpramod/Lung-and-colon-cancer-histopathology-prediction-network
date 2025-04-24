# interface.py
# Standardizes names for grading as per PDF [cite: 27, 28, 29, 30, 31, 32, 33]

print("Loading interface...")

# --- Model Import ---
try:
    # Ensure 'HistopathModel' is the correct class name in model.py
    from model import HistopathModel as TheModel
    print(f"  OK: Imported model.HistopathModel as TheModel")
except ImportError as e:
    print(f"  ERROR importing model: {e}")
    TheModel = None

# --- Training Function Import ---
try:
    # Ensure 'train_model' is the correct function name in train.py
    from train import train_model as the_trainer
    print(f"  OK: Imported train.train_model as the_trainer")
except ImportError as e:
    print(f"  ERROR importing trainer function: {e}")
    the_trainer = None

# --- Prediction Function Import ---
try:
    # Ensure 'predict_batch' is the correct function name in predict.py
    from predict import predict_batch as the_predictor
    print(f"  OK: Imported predict.predict_batch as the_predictor")
except ImportError as e:
    print(f"  ERROR importing predictor function: {e}")
    the_predictor = None

# --- Dataset Class Import ---
try:
    # Ensure 'HistopathDataset' is the correct class name in dataset.py
    from dataset import HistopathDataset as TheDataset
    print(f"  OK: Imported dataset.HistopathDataset as TheDataset")
except ImportError as e:
    print(f"  ERROR importing dataset class: {e}")
    TheDataset = None

# --- DataLoader Function Import ---
# NOTE: PDF [cite: 32] shows importing the loader itself ('unicornLoader').
# Here we import the function that *creates* the loader ('create_dataloader').
# This might need adjustment based on the exact expectation of the grading script.
try:
    # Ensure 'create_dataloader' is the correct function name in dataset.py
    from dataset import create_dataloader as the_dataloader
    print(f"  OK: Imported dataset.create_dataloader as the_dataloader")
except ImportError as e:
    print(f"  ERROR importing dataloader function: {e}")
    the_dataloader = None

# --- Config Variable Imports ---
try:
    # Ensure 'BATCH_SIZE' exists in config.py
    from config import BATCH_SIZE as the_batch_size
    print(f"  OK: Imported config.BATCH_SIZE as the_batch_size (Value: {the_batch_size})")
except ImportError as e:
    print(f"  ERROR importing batch size from config: {e}")
    the_batch_size = None

try:
    # Ensure 'NUM_EPOCHS' exists in config.py
    from config import NUM_EPOCHS as total_epochs
    print(f"  OK: Imported config.NUM_EPOCHS as total_epochs (Value: {total_epochs})")
except ImportError as e:
    print(f"  ERROR importing num epochs from config: {e}")
    total_epochs = None

print("\nInterface loading complete. Check for ERROR messages above.")
# --- Final Check ---
if None in [TheModel, the_trainer, the_predictor, TheDataset, the_dataloader, the_batch_size, total_epochs]:
     print("\n*** WARNING: One or more required components failed to import correctly via interface.py. ***")
     print("*** The grading script will likely fail. Please fix the import errors listed above. ***\n")
else:
     print("\nAll required interface components seem to be imported successfully.")