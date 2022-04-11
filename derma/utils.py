from multiprocessing.connection import wait
import torch
from tqdm import tqdm
import numpy as np

from derma.metric import Metrics

def train(model, train_loader, optimizer, criterion, n_epoch, tb_writer, val_loader=None):
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
        for idx, (inputs, targets) in enumerate(train_loader):
            model.train()
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

# INSPIRED IN:  
# https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/
def train_val(model,trainloader,validloader,optimizer,criterion,tb_writer,patience=10,epochs=10,model_dir='saved_model.pth'):
    import numpy as np

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device) 

    min_valid_loss = np.inf

    epoch_iterator = tqdm(
        range(epochs),
        leave=True,
        unit="epoch",
        postfix={"tls": "%.4f" % 1,
                "tacc": "%.4f" % 0,
                "vls": "%.4f" % 1,
                "vacc": "%.4f" % 0
        },
    )
    # Metrics
    loss_tr = []
    loss_vl = []    
    step_tr=0
    step_vl=0
    another_epoch = 0

    for e in epoch_iterator:
        ### TRAIN ###
        model.train()     # Optional when not using Model Specific layer
        for idx,(inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            train_loss = loss.item()
            loss_tr.append(train_loss)
            loss.backward()
            optimizer.step()
            if idx % 150 == 0:
                # Accuracy
                accuracy = Metrics.accuracy(outputs,targets)

                epoch_iterator.set_postfix(
                    tls="%.4f" % np.mean(loss_tr),
                    tacc="%.4f" % accuracy
                )
                tb_writer.add_scalar('train/loss', np.mean(loss_tr), step_tr)
                tb_writer.add_scalar('train/accuracy', accuracy, step_tr)
                loss_tr = []
                step_tr+=1
        ### TEST ###
        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for idx,(inputs, targets) in enumerate(validloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            valid_loss = loss.item() * inputs.size(0)
            loss_vl.append(valid_loss)
            if idx % 150 == 0:
                val_accuracy = Metrics.accuracy(outputs,targets)
#                epoch_iterator.set_postfix(
#                    vls="%.4f" % np.mean(loss_vl),
#                    vacc="%.4f" % val_accuracy
#                )
                tb_writer.add_scalar('test/accuracy', val_accuracy, step_vl)
                tb_writer.add_scalar('test/loss', np.mean(loss_vl), step_vl)
                loss_vl = []
                step_vl+=1

#        print(f'Epoch {e+1} \n Training Loss: {train_loss / len(trainloader)} \n Validation Loss: {valid_loss / len(validloader)}')
        if min_valid_loss >= valid_loss:
#            print(f'Validation Loss Decreased({min_valid_loss:.6f} ---> {valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), model_dir)
        else:
            another_epoch += 1
            if patience < another_epoch:
                print('Enough epochs. Training is stopping.')
                break
