import torch


def NLL_loss_function(pred, target):
    # Reduce the predictions to a single value by averaging
    pred_mean = pred.mean(dim=1)
    variance = torch.var(pred, dim=1, unbiased=False)
    variance = torch.clamp(variance, min=0.1)
    delta_y = torch.abs(pred_mean - target)
    loss = 0.5 * (torch.log(2 * torch.pi * variance) + (delta_y ** 2) / variance)
    return loss.mean(), variance.mean()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss=0
    total_variance=0
    for data in train_loader:
        data = data.to(device)  # Move data to GPU
        optimizer.zero_grad()  # Zero the gradients
        pred = model(data)  #model(data[i].z, data[i].pos, data[i].batch)
        loss, variance = NLL_loss_function(pred, data.y)  # Compute the loss for the entire dataset
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the weights
        total_loss += loss.item()
        total_variance += variance.item() 
    return total_loss / len(train_loader), total_variance / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    total_loss= 0
    total_variance=0
    with torch.no_grad():
        for data in (test_loader):
            data = data.to(device) 
            val_pred = model(data)
            val_loss, val_variance = NLL_loss_function(val_pred, data.y)  # Compute the validation loss
            total_loss += val_loss.item()
            total_variance += val_variance.item() 
    return total_loss / len(test_loader), total_variance / len(test_loader)

def adjust_learning_rate(optimizer, current_step, config):
    """
    Adjust the learning rate based on the current step, warmup steps, and milestones.

    Parameters:
    - optimizer: the optimizer whose learning rate needs to be adjusted
    - current_step: the current training step
    - config: a dictionary containing learning rate configuration with keys:
      - 'lr_initial': initial learning rate
      - 'lr_gamma': factor by which the learning rate will be multiplied at each milestone
      - 'lr_milestones': list of steps at which learning rate decays
      - 'warmup_steps': number of steps for warmup
      - 'warmup_factor': initial factor during warmup
    """
    lr_initial = config['lr_initial']
    lr_gamma = config['lr_gamma']
    lr_milestones = config['lr_milestones']
    warmup_steps = config['warmup_steps']
    warmup_factor = config['warmup_factor']
    
    # Warmup phase
    if current_step < warmup_steps:
        lr = lr_initial * (warmup_factor + (1 - warmup_factor) * (current_step / warmup_steps))
    else:
        # Learning rate decay phase (after warmup)
        lr = lr_initial
        for milestone in lr_milestones:
            if current_step >= milestone:
                lr *= lr_gamma
            else:
                break
    
    # Update the learning rate for each parameter group in the optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class EarlyStopping:
    def __init__(self, patience: int = 20, delta: float = 0):
        """
        Initialize EarlyStopping.

        Parameters:
        - patience: Number of epochs to wait for improvement.
        - delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss: float):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True