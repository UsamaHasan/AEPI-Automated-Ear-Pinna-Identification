from torch.utils.data import Dataset
from torchvision.utils import make_grid

class CustomDataset(Dataset):
    """
    A Custom dataset class to load numpy images and apply torchvision.transforms on them.s 
    """
    def __init__(self , tensors , transform = None ):   
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self,index):
        x = self.tensors[0][index]  
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x)
        return x , y

    def __len__(self):
        return len(self.tensors[0]) 

def batch_viz(batch, one_channel=True):
    plt.figure(figsize=(8,8))
    grid = make_grid(batch,padding=2,normalize=True)
    plt.imshow(np.transpose(grid,(1,2,0)))

def init_weights(m):
    """"""
    if isinstance(m,(nn.Conv2d ,nn.Linear )) :
        torch.nn.init.kaiming_normal_(m.weight)