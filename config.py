import torch

DATASET_ROOT="Dataset"

CLASSES = ['lung_aca', 'lung_n', 'lung_scc', 'colon_aca', 'colon_n']
NUM_CLASSES = 5  # lung_aca, lung_n, lung_scc, colon_aca, colon_n

TEST_SPLIT=0.2
VAL_SPLIT=0.2

IMG_SIZE=64
INPUT_CHANNELS=3

DATASET_MEAN = [0.7290357947349548, 0.5999864339828491, 0.8766909241676331]
DATASET_STD = [0.16132670640945435, 0.1930888444185257, 0.08985170722007751]


MODEL_NAME = "histopath_cnn_cleaned_v1"
BATCH_SIZE = 64      
NUM_EPOCHS = 100    
LEARNING_RATE = 0.00015 
WEIGHT_DECAY = 1e-4   # Optimizer weight decay
SCHEDULER_STEP = 20   # LR scheduler step size in epochs
SCHEDULER_GAMMA = 0.5 # LR scheduler reduction factor


NUM_WORKERS = 2 

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
RANDOM_SEED = 42


PATIENCE = 10

CHECKPOINT_DIR = "_checkpoints"

FINAL_WEIGHTS_FILE = "final_weights.pth"

BEST_VAL_WEIGHTS_FILE = "best_val_weights.pth"