from torch.utils.data import Dataset
import numpy as np
import os
from torch import LongTensor, from_numpy, Generator

def get_samples_weight(dataset,print_results=False):
    from torch import from_numpy
    y = [dataset[i][1] for i in range(len(dataset))]
    class_sample_count = np.array([len(np.where(y == t)[0]) for t in np.unique(y)])
    weight = 1. / class_sample_count
    samples_weight = from_numpy(np.array([weight[t] for t in y])).flatten().double()
    if print_results:
        print('Samples per class: {}'.format(class_sample_count))
        print('Weight per class: {}'.format(weight))
    return samples_weight

class Derma(Dataset):
    def __init__(self, root_dir: str, labels=[0, 1], transform=None) -> None:
        super(Derma, self).__init__()
        '''
            Definir tu conjunto de datos tal que X e Y estén "relacionadas"

            1. Buscar en el directorio y almacenar las rutas
        '''
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

    def split_det(self,split_ratio=0.9):
        from torch import Tensor, split
        split_sizes = (int(split_ratio * self.__len__()), self.__len__() - int(split_ratio * self.__len__()))
        train_set, test_set = split(Tensor(list(zip(self.x,self.y))), split_sizes)
        return train_set, test_set
    
    def split_rand(self,split_ratio=0.9,manual_seed=None):
        import random
        from torch.utils.data import random_split
        from torch import split
        if manual_seed:
            seed = manual_seed
        else:
            seed = random.random()
        generator = Generator().manual_seed(seed)
        split_sizes = (int(split_ratio * self.__len__()), self.__len__() - int(split_ratio * self.__len__()))
        train_set, test_set = random_split(self, split_sizes,generator=generator)
        return train_set, test_set