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
    train(model,[train_loader, val_loader], optimizer, criterion, n_epoch, tb_writer)
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
    return metrics_batch, labels_batch, output_batch

def load_experiments_results(test,DB_used):
    from config import DATASET_DIR, RESULT_DIR
    from derma.dataset import get_samples_weight
    from derma.experiment import test_experiment
    inverted_residual_setting_v0 = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ] #ORIGINAL
    inverted_residual_setting_vT3 = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 1, 2],
            [4, 32, 1, 2],
            [4, 64, 1, 2],
            [4, 96, 1, 1],
            [4, 160, 1, 2],
            [4, 320, 1, 1],
        ]
    batch_size = 224
    if test == 1:
        Test = 'MbV2'
        CoordAtt = False
        inverted_residual_setting = inverted_residual_setting_v0
    elif test == 2:
        Test = 'MbV2_CA'
        CoordAtt = True
        inverted_residual_setting = inverted_residual_setting_v0
    elif test == 3:
        Test = 'MbV2_CA_Reduced'
        CoordAtt = True
        inverted_residual_setting = inverted_residual_setting_vT3
    elif test == 4:
        Test = 'MbV2_Reduced'
        CoordAtt = False
        inverted_residual_setting = inverted_residual_setting_vT3
    if DB_used == 'HAM10000' or DB_used == 'HAM_test19':
        dataset_dir = os.path.join(DATASET_DIR,'HAM10000_splited')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
    elif DB_used == 'HAM_norm':
        dataset_dir = os.path.join(DATASET_DIR,'HAM10000_splited')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif DB_used == 'ISIC2019' or DB_used == 'ISIC19_testHAM':
        dataset_dir = os.path.join(DATASET_DIR,'ISIC2019_splited')
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif DB_used == 'PH2':
        dataset_dir = os.path.join(DATASET_DIR,'PH2_up_splited')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        batch_size = 6
    if DB_used == 'HAM_test19':
        model_dir = os.path.join(RESULT_DIR,'weights',Test,'HAM10000','model.pth')
        test_set = Derma(os.path.join(DATASET_DIR,'ISIC2019_balanced_000_256'),labels=[0,1],transform=transform)
        print(model_dir)
        print(os.path.join(DATASET_DIR,'ISIC2019_balanced_000_256'))
    elif DB_used == 'ISIC19_testHAM':
        model_dir = os.path.join(RESULT_DIR,'weights',Test,'ISIC2019','model.pth')
        test_set = Derma(os.path.join(DATASET_DIR,'HAM10000_balanced_000'),labels=[0,1],transform=transform)
        print(model_dir)
        print(os.path.join(DATASET_DIR,'HAM10000_balanced_000'))
    else:
        model_dir = os.path.join(RESULT_DIR,'weights',Test,DB_used,'model.pth')
        test_set = Derma(os.path.join(dataset_dir,'test'),labels=[0,1],transform=transform)

#    test_sampler, test_weights = get_samples_weight(test_set,print_results=False)
#    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, sampler=test_sampler, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0, shuffle=False)
    if CoordAtt:
        model = MobileNetV2(num_classes=2, inverted_residual_setting=inverted_residual_setting, block=InvertedResidual)
    else:
        model = MobileNetV2(num_classes=2, inverted_residual_setting=inverted_residual_setting) # standard MobileNetV2
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.load_state_dict(torch.load(model_dir))
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_dir),map_location=torch.device('cpu'))
    test_result, _, _ = test_experiment(model,test_loader)
    experiment_name = DB_used + '_' + Test
    return test_result, experiment_name

