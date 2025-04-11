# QM9-SE model config.py

# General Hyperparameters
batch_size = 100
num_epochs = 1000
lr = 5e-5


# Model Configuration
model_config = {
    'hidden_channels': 128,
    'num_filters': 128,
    'num_interactions': 6,
    'num_gaussians': 50,
    # 'atomref': dataset.atomref(7)  ← needs to be set in the main script when dataset is loaded
}

# Loss Target (Internal Energy)
target_index = 7  # data.y[:, 7]

# Learning Rate Scheduler Settings
lr_scheduler_config = {
    'factor': 0.8,
    'patience': 25,
    'min_lr': 1e-6
}

# Checkpoints (Training from scratch)
save_path = '../checkpoints/QM9-SE.pth'

