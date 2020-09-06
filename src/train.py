import torch
import numpy as np
from models import Net
import torch.optim as optim
from sklearn.model_selection import train_test_split
from utils.train_utils import train_model
from torch.utils.data import DataLoader , TensorDataset
from utils.utils import CustomDataset
from tqdm import tqdm



if __name__ == '__main__':

    dataset = np.load('..dataset\images_PIL.npy',allow_pickle=True,encoding='bytes') #load dataset file
    trainX , validX , trainY , validY = train_test_split(dataset,labels,test_size=0.2,random_state = 44)
    trainY = torch.tensor(trainY).long()
    validY = torch.tensor(validY).long()
    trainY-=1
    validY-=1
    train_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                      transforms.Resize((80,80)),
                                      transforms.RandomCrop(64,padding=4),
                                      transforms.ColorJitter(brightness=0.5),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4601,0.4601,0.4601],[0.2701,0.2701,0.2701])])
    valid_transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                        transforms.Resize((64,64)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4556,0.4556,0.4556],[0.2716,0.2716,0.2716])
                                        ])
    trainDataset = CustomDataset((trainX,trainY),train_transform)
    validDataset = CustomDataset((validX,validY),valid_transform)
    trainLoader = DataLoader(trainDataset,batch_size = 64 , num_workers=4,shuffle=True)
    validationLoader  = DataLoader(validDataset, batch_size=32 , num_workers=4,shuffle=True)
    batch , labels = next(iter(trainLoader))
    print(f'Batch` Size:{batch.size()}')
    
    NUM_EPOCH = 50
    MODEL_PATH = 'final_model'
    FINAL_ACCURACY = 0.0 
    NUM_CLASSES = 164

    # Initializing model
    criterion = torch.nn.CrossEntropyLoss()
    clip_value = 0.5
    model = Net()
    model = model.to(device)
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
    optimizer = optim.Adam(model.parameters() , lr = 1e-3)
    model_fit = train_model(model ,criterion = criterion, optimizer= optimizer ,NUM_EPOCHS=NUM_EPOCH)