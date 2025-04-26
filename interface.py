print("Loading interface...")

try:
    from model import HistopathModel as TheModel
    print(f"  OK: Imported model.HistopathModel as TheModel")
except ImportError as e:
    print(f"  ERROR importing model: {e}")
    TheModel = None

try:
    from train import train_model as the_trainer
    print(f"  OK: Imported train.train_model as the_trainer")
except ImportError as e:
    print(f"  ERROR importing trainer function: {e}")
    the_trainer = None

try:
    from predict import predict_batch as the_predictor
    print(f"  OK: Imported predict.predict_batch as the_predictor")
except ImportError as e:
    print(f"  ERROR importing predictor function: {e}")
    the_predictor = None

try:
    from dataset import HistopathDataset as TheDataset
    print(f"  OK: Imported dataset.HistopathDataset as TheDataset")
except ImportError as e:
    print(f"  ERROR importing dataset class: {e}")
    TheDataset = None

try:
    from dataset import create_dataloader as the_dataloader
    print(f"  OK: Imported dataset.create_dataloader as the_dataloader")
except ImportError as e:
    print(f"  ERROR importing dataloader function: {e}")
    the_dataloader = None

try:
    from config import BATCH_SIZE as the_batch_size
    print(f"  OK: Imported config.BATCH_SIZE as the_batch_size (Value: {the_batch_size})")
except ImportError as e:
    print(f"  ERROR importing batch size from config: {e}")
    the_batch_size = None

try:
    from config import NUM_EPOCHS as total_epochs
    print(f"  OK: Imported config.NUM_EPOCHS as total_epochs (Value: {total_epochs})")
except ImportError as e:
    print(f"  ERROR importing num epochs from config: {e}")
    total_epochs = None

print("\nInterface loading complete.")

