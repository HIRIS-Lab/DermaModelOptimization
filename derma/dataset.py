from torch.utils.data import Dataset
from PIL import Image

class Derma(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        super(Derma, self).__init__()
        '''
            Definir tu conjunto de datos tal que X e Y estén "relacionadas"

            1. Buscar en el directorio y almacenar las rutas
        '''
        
        # self.x = ['test/0.png', 'img/1.png', 'img/2.png', 'img/3.png']
        # self.y = [0, 1, 0, 1]
        self.x = []
        self.y = []

        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = Image.open(self.x[idx]).convert('RGB')
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)

        return (x, y)