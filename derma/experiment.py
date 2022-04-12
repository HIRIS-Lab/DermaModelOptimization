import os, sys
project_dir = os.path.join(os.getcwd(),'..')
if project_dir not in sys.path:
    sys.path.append(project_dir)

attention_dir = os.path.join(project_dir, 'modules/AttentionMap')
if attention_dir not in sys.path:
    sys.path.append(attention_dir)

sparse_dir = os.path.join(project_dir, 'modules/Sparse')
if sparse_dir not in sys.path:
    sys.path.append(sparse_dir) 

from derma.dataset import Derma
from derma.architecture import InvertedResidual
from derma.utils import train, train_val

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.models import MobileNetV2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def problem_def(dataset_dir,Weighted_sampling,labels=[0,1],split_ratio=[0.8,0.1,0.1],transform=None,batch_size=32):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    # Derma dataset 
    dataset = Derma(dataset_dir,labels=labels,transform=transform)
    # Train-test splitting
    #dataset.shuffle(manual_seed=42) 
    train_set, val_set, test_set = dataset.split_rand(split_ratio=split_ratio,manual_seed=42)
    # Weighted sampling
    if Weighted_sampling:
        from derma.dataset import get_samples_weight
        train_sampler = get_samples_weight(train_set)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=train_sampler)
        val_sampler = get_samples_weight(val_set)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0, sampler=val_sampler)
        test_sampler = get_samples_weight(test_set)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, sampler=test_sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader, test_loader

def train_experiment(problem_name,train_loader, val_loader,CoordAtt,inverted_residual_setting,num_classes=2,criterion=torch.nn.CrossEntropyLoss(),n_epoch=10):
    if CoordAtt:
        model = MobileNetV2(num_classes=num_classes, inverted_residual_setting=inverted_residual_setting, block=InvertedResidual)
    else:
        model = MobileNetV2(num_classes=num_classes, inverted_residual_setting=inverted_residual_setting) # standard MobileNetV2
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    tb_writer = SummaryWriter(log_dir=os.path.join('log',problem_name))
    model_dir = os.path.join('models',problem_name+'.pt')
    train_val(model,train_loader,val_loader,optimizer,criterion,tb_writer,n_epoch,model_dir=model_dir)
    torch.save(model.state_dict(),model_dir)

def load_experiment(model,model_dir):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_dir))
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_dir), map_location=torch.device('cpu'))

def metrics(output: torch.Tensor, target: torch.Tensor,weights=None):
    from derma.metric import Metrics
    import pandas as pd
    acc = Metrics.accuracy(output, target)
    sensitivity, specificity, precission, recall = Metrics.performance(output,target,weights=weights)
    var = [[sensitivity, specificity, precission, recall, acc]]
    columns = ['Sensitivity', 'Specificity', 'Precission', 'Recall', 'Accuracy']
    metrics = pd.DataFrame(var, columns=columns)
    return metrics

def test_experiment(model,dataloader,weights=None):
    from torch.autograd import Variable
#    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = model.to(device) 
    # set model to evaluation mode
    model.eval()
    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch).to(device), Variable(labels_batch).to(device)
        # compute model output
        output_batch = model(data_batch)
        metrics_batch = metrics(output_batch,labels_batch,weights=weights)
    return metrics_batch