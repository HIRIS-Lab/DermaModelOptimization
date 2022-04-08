import torch
from tqdm import tqdm
import numpy as np

def train(model, loader, optimizer, criterion, n_epoch, tb_writer):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device) 

    epoch_iterator = tqdm(
        range(n_epoch),
        leave=True,
        unit="epoch",
        postfix={"tls": "%.4f" % 1},
    )

    # Metrics
    loss_tr = []
    total = 0
    correct = 0
    
    step=0

    for _ in epoch_iterator:
        for idx, (inputs, targets) in enumerate(loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_tr.append(loss.item())

            loss.backward()
            optimizer.step()

            # Accuracy
            predict = torch.argmax(outputs, 1)
            total += targets.size(0)
            correct += torch.eq(predict, targets).sum().double().item()

            if idx % 150 == 0:
                accuracy = correct/total

                epoch_iterator.set_postfix(
                    tls="%.4f" % np.mean(loss_tr),
                    acc="%.4f" % accuracy
                )
                
                tb_writer.add_scalar('train/loss', np.mean(loss_tr), step)
                tb_writer.add_scalar('train/accuracy', accuracy, step)
                step+=1

                loss_tr = []
                total = 0
                correct = 0

def test(model, loader, tb_writer):
    import tqdm
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device) 

    n_epoch = 1
    epoch_iterator = tqdm(
        range(n_epoch),
        leave=True,
        postfix={"tls": "%.4f" % 1},
    )
#        unit="epoch",

    # Metrics
    loss_tr = []
    total = 0
    correct = 0
    step=0

    for _ in epoch_iterator:
        for idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Accuracy
            predict = torch.argmax(outputs, 1)
            total += targets.size(0)
            correct += torch.eq(predict, targets).sum().double().item()

            if idx % 150 == 0:
                accuracy = correct/total

                epoch_iterator.set_postfix(
                    tls="%.4f" % np.mean(loss_tr),
                    acc="%.4f" % accuracy
                )
                
                tb_writer.add_scalar('test/loss', np.mean(loss_tr), step)
                tb_writer.add_scalar('test/accuracy', accuracy, step)
                step+=1

                loss_tr = []
                total = 0
                correct = 0