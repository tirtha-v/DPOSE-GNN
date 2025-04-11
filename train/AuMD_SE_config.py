# AuMD-SE model config.py

# General Training Hyperparameters
batch_size = 20
eval_batch_size = 8
num_epochs = 500
learning_rate = 0.0001


# Model Configuration
model_config = {
    'hidden_channels': 1024,
    'num_filters': 256,
    'num_interactions': 5,
    'num_gaussians': 200,
    'num_targets': 64,
    'cutoff': 6.0
}

# Learning Rate Scheduler Settings 
lr_scheduler_config = {
    'mode': 'min',
    'factor': 0.8,
    'patience': 25,
    'min_lr': 1e-6
}


# EarlyStopping Settings 
early_stopping_config = {
    'patience': 70,
    'delta': 0.000001
}


# Checkpoint
checkpoint_path = '../checkpoints/schnet_ocp_checkpoint.pt'
save_path = '../checkpoints/AuMD-SE.pth'