import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_data, get_data_loaders
from model import initialize_model
from optimizers.adam import train_adam
from optimizers.rmsprop import train_rmsprop
from optimizers.adamw import train_adamw

# Hyperparameters
BATCH_SIZE = 32
k_folds = 5  # Number of folds for cross-validation

# Load dataset
trainset, _ = get_data()
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize model
model = initialize_model()

# Define hyperparameter search space for each optimizer
adam_hyperparams = {
    'learning_rates': [0.0001, 0.001, 0.01],
    'betas': [(0.9, 0.999), (0.85, 0.98), (0.95, 0.999)],
    'weight_decays': [0.0001, 0.001, 0.01],
    'epochs': 5
}

rmsprop_hyperparams = {
    'learning_rates': [0.0001, 0.001, 0.01],
    'alphas': [0.85, 0.9, 0.99],
    'epochs': 5
}

adamw_hyperparams = {
    'learning_rates': [0.0001, 0.001, 0.01],
    'betas': [(0.9, 0.999), (0.85, 0.98), (0.95, 0.999)],
    'weight_decays': [0.0001, 0.001, 0.01],
    'epochs': 5
}

# Train using different optimizers
print("\nTraining with Adam")
train_adam(model, trainset, kf, adam_hyperparams, BATCH_SIZE)

print("\nTraining with RMSprop")
train_rmsprop(model, trainset, kf, rmsprop_hyperparams, BATCH_SIZE)

print("\nTraining with AdamW")
train_adamw(model, trainset, kf, adamw_hyperparams, BATCH_SIZE)
