from torch.utils.data import Dataset
import numpy as np
import os
from torch import LongTensor

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
