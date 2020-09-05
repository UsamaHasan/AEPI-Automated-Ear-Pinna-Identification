import torch
import numpy as np
from model import Net
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from utils import to_categorical
from torch.utils.data import DataLoader , TensorDataset

num_classes = 5

if __name__ == '__main__':

    #Data Loading.
    dataset = np.load('images.npy',encoding='bytes') #load dataset file
    labels  = np.load('labels.npy') #load labels file
    dict_label = {36:0, 62:2, 97:3, 49:1, 157:4}
    print(dict_label)
    
    new_labels = []
    for label in labels:
        new_labels.append(dict_label[label])
    
    x_train , x_test , y_train , y_test = train_test_split(dataset,new_labels,test_size= 0.33)
    
    #Channel first for PyTorch
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[3],x_train.shape[1],x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[3],x_test.shape[1],x_test.shape[2])
    
    print('Train Data Shape',x_train.shape)
    print('Test Data Shape',x_test.shape)

    #Converting datatype to Float32
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    #Normalizing the Input
    x_train/=255
    x_test/=255

    #Converting labels to OneHot vector
    #y_train = to_categorical(y_train)

    y_train = np.array(y_train)
 #   y_train = y_train - 1
  #  y_test = y_test - 1
    
    
    #Converting Numpy array into Tensor
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train).long()
    print(f'Tensor Y:{tensor_y.size()}')
    dataset = TensorDataset(tensor_x , tensor_y)
    trainloader = DataLoader(dataset , batch_size = 16, shuffle=True)

    #Initializing Model.
    
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net = Net(num_classes=num_classes)
    net.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(net.parameters(), lr = 0.01 , momentum=0.9 )
    
    #Training Mini-batches.
    
    for epoch in range(100):
        running_loss = 0.0
        for i, data in enumerate(trainloader , 0):
            images , labels = data
            
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)

            labels = labels.to(device)
            images = images.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            print(f'Label: {labels}')
            print(f'Label Shape:{labels.size()} , Ouput Shape:{outputs.size()}') 
            loss = criterion(outputs , labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            
            running_loss += loss.item()
            if i % 300 == 299:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 300))
                running_loss = 0.0

    print('Finished Training')

