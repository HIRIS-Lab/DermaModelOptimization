import torch, copy
from tqdm import tqdm
import numpy as np
from .metric import Metrics

def train(model, loader, optimizer, criterion, n_epoch, tb_writer, reconstruction=False) -> float:
    '''
    Train the model

    Args:
        model: nn.Module,
            Model to train
        
        loader: DataLoader or list,
            If list, then it should be a list of two DataLoader: train and validation loader.
            It is necessary if you want to use validation set for a early stopping approach,
            saving the model which best works for validation set.
        
        optimizer: optim.Optimizer,
            Optimizer to use during training.
            
        criterion: loss function
        n_epoch: number of epochs to train
        tb_writer: tensorboard writer
        reconstruction: Indicate it is a reconstruction problem.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    train_loader, val_loader = loader if isinstance(loader, list) else (loader, None)
    phase = ['train', 'eval'] if val_loader is not None else ['train']

    epoch_iterator = tqdm(
        range(n_epoch),
        leave=True,
        unit="epoch",
        postfix={"tls": "%.4f" % 1},
    )

    for epoch in epoch_iterator:
        for p in phase:
            model.train() if p == 'train' else model.eval()
            loader = train_loader if p == 'train' else val_loader

            epoch_acc_sum = 0.
            epoch_loss_sum = 0.
            for idx, (inputs, targets) in enumerate(loader):
                optimizer.zero_grad()

                inputs = inputs.to(device)
                targets = targets.to(device) if not reconstruction else inputs

                with torch.set_grad_enabled(p == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
    
                    epoch_loss_sum += loss.item()
                    epoch_acc_sum += Metrics.accuracy(outputs, targets) if not reconstruction else 0

                    if p == 'train':
                        loss.backward()
                        optimizer.step()

                        if idx % 100 == 0:
                            epoch_iterator.set_postfix(
                                tls="%.4f" % (epoch_loss_sum/(idx+1)),
                                acc="%.4f" % (epoch_acc_sum/(idx+1))
                            )

            epoch_loss = epoch_loss_sum/(idx+1)
            epoch_acc = epoch_acc_sum/(idx+1)
            tb_writer.add_scalar('{}/loss'.format(p), epoch_loss, epoch)
            if not reconstruction:
                tb_writer.add_scalar('{}/accuracy'.format(p), epoch_acc, epoch)

            if (p == 'eval') and (epoch_loss < best_loss):
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                    
    if val_loader is not None:
        # load best model weights if validation was done
        model.load_state_dict(best_model_wts)
    
    return best_loss