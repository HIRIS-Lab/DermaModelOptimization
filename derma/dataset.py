from logging import warning
from torch.utils.data import Dataset
import numpy as np
import os
from torch import LongTensor, from_numpy, Generator

class Derma(Dataset):
    def __init__(self, root_dir: str, labels=[0, 1], transform=None) -> None:
        super(Derma, self).__init__()
        self.x = []
        self.y = []
        for label in np.unique(labels):
            self.x = self.x + [os.path.join(root_dir,str(label),name) for name in os.listdir(os.path.join(root_dir,str(label)))]
            self.y = self.y + [label]*len(os.listdir(os.path.join(root_dir,str(label))))
        self.y = LongTensor(self.y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        from PIL import Image
        x = Image.open(self.x[idx]).convert('RGB')
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return (x, y)

    def get_labels(self):
        return self.y

    def shuffle(self,manual_seed=None):
        import random
        if manual_seed:
            seed = manual_seed
        else:
            seed = random.random()
        random.seed(seed)
        dummy = list(zip(self.x,self.y))
        random.shuffle(dummy)
        self.x, self.y = zip(*dummy)
        return self

    def split_det(self,split_ratio):
        from torch import Tensor, split
        train_size = int(split_ratio[0]*self.__len__())
        val_size = int(split_ratio[1]*self.__len__())
        test_size = self.__len__() - train_size - val_size
        train_set, val_set, test_set = split(Tensor(list(zip(self.x,self.y))), (train_size,val_size,test_size))
        return train_set, val_set, test_set
    
    def split_rand(self,split_ratio,manual_seed=None):
        import random
        from config import RANDOM_SEED
        from torch.utils.data import random_split
        if manual_seed:
            seed = manual_seed
        else:
            seed = RANDOM_SEED
        generator = Generator().manual_seed(seed)
        train_size = int(split_ratio[0]*self.__len__())
        val_size = int(split_ratio[1]*self.__len__())
        test_size = self.__len__() - train_size - val_size
        train_set, val_set, test_set = random_split(self,(train_size,val_size,test_size),generator=generator)
        return train_set, val_set, test_set
    
    def split_labels(self,split_ratio,manual_seed=None):
        import random
        from config import RANDOM_SEED
        from torch.utils.data import random_split
        if manual_seed:
            seed = manual_seed
        else:
            seed = RANDOM_SEED
        generator = Generator().manual_seed(seed)
        train_size = int(split_ratio[0]*self.__len__())
        val_size = int(split_ratio[1]*self.__len__())
        test_size = self.__len__() - train_size - val_size
        y_test, y_val, y_test = random_split(self.y,(train_size,val_size,test_size),generator=generator)
        return y_test, y_val, y_test

def get_samples_weight(dataset,y=None,print_results=True):
    from torch import from_numpy
    from torch.utils.data import WeightedRandomSampler
    from config import RANDOM_SEED
    if y is None:
        y = [dataset[i][1] for i in range(len(dataset))]
    class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = from_numpy(np.array([weight[t] for t in y])).flatten().double()
    if print_results:
        print('Samples per class: {}'.format(class_sample_count))
        print('Weight per class: {}'.format(weight))
    generator = Generator().manual_seed(RANDOM_SEED)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True,generator=generator)
    return sampler, samples_weight