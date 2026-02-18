"""
Configuration for CIFAR-10 classification experiment.
Edit this file to change hyperparameters.
"""

# Dataset
DATASET = "CIFAR-10"
NUM_CLASSES = 10
DATA_DIR = "./data"
# Use a subset of training data to keep ~5-10s/epoch.
# None = full 50K. Set to e.g. 10000 for faster experiments.
TRAIN_SUBSET = 10000

# Model
MODEL = "SmallCNN"

# Training
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adam"       # "adam" or "sgd"
LR_SCHEDULE = "none"     # "none", "cosine", "step"

# Checkpointing
CHECKPOINT_DIR = "./checkpoints"
CHECKPOINT_FILE = "./checkpoints/best.pt"
METRICS_FILE = "./results.json"
