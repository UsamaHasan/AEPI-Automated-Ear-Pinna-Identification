from torch.utils.data import Dataset
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

def matplotlib_imshow(img, one_channel=True):
    """
    To plot tensor as images 
    """
    npimg = img.detach().numpy()
    if one_channel:
        npimg = npimg.reshape(64,32)
        plt.imshow(npimg, cmap="Greys")
        plt.show()
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))