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
from derma.utils import train

import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.models import MobileNetV2
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

def train_experiment(dataset_dir,problem_name,Weighted_sampling,CoordAtt,inverted_residual_setting,
                    labels=[0,1],split_ratio=0.9,transform=None,batch_size=32,criterion=torch.nn.CrossEntropyLoss(),n_epoch=10):

    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    log_dir = os.path.join('log',problem_name) 
    model_dir = os.path.join('models',problem_name+'.pt')

    # Derma dataset 
    dataset = Derma(dataset_dir,labels=labels,transform=transform)

    # Train-test splitting
    #dataset.shuffle(manual_seed=42) 
    train_set, test_set = dataset.split_rand(split_ratio=split_ratio,manual_seed=42)

    # Weighted sampling
    if Weighted_sampling:
        from derma.dataset import get_samples_weight
        samples_weight = get_samples_weight(train_set)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

        # Data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, sampler=sampler)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    # ISIC2018
    if CoordAtt:
        model = MobileNetV2(num_classes=len(labels), inverted_residual_setting=inverted_residual_setting, block=InvertedResidual)
    else:
        model = MobileNetV2(num_classes=len(labels), inverted_residual_setting=inverted_residual_setting) # standard MobileNetV2

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    tb_writer = SummaryWriter(log_dir=log_dir)

    train(model, train_loader, optimizer, criterion, n_epoch, tb_writer)

    torch.save(model.state_dict(), model_dir)

    return test_loader


def load_experiment(model,model_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(model_dir))
        model.to(device)
        model.eval()
    else:
        model.load_state_dict(torch.load(model_dir), map_location=torch.device('cpu'))
        model.eval()

def test_experiment():
    # por definir

#from derma.utils import test
#from torch.utils.tensorboard import SummaryWriter

#test(model, test_loader, tb_writer)

# POR DEFINIR
#model.eval()
#output = model(input)

    results = ''
    return results