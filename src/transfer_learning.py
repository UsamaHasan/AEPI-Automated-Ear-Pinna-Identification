import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader , TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split


NUM_EPOCH = 20
MODEL_PATH = 'final_model'
FINAL_ACCURACY = 0.0 
NUM_CLASSES = 164

if __name__ == '__main__':

    #Data Loading.
    dataset = np.load('images.npy',encoding='bytes') #load dataset file
    labels  = np.load('labels.npy') #load labels file
    x_train , x_test , y_train , y_test = train_test_split(dataset,labels,test_size= 0.33)
    
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
    y_train = y_train - 1
    y_test = y_test - 1
    
    #Converting Numpy array into Tensor
    x_train = torch.Tensor(x_train)
    x_test  = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train).long()
    y_test  = torch.Tensor(y_test).long()

    #Train Loader
    train_dataset = TensorDataset(x_train , y_train)
    trainloader = DataLoader(train_dataset , batch_size = 8, shuffle=True,num_workers=4 )

    # Test Loader
    test_dataset = TensorDataset(x_test , y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4 )

    # Initializing model
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    model =  models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters() , lr = 0.001 , momentum = 0.1)
    
    best_accuracy = 0.0

    for epoch in  range(NUM_EPOCH):
        for images , labels in iter(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs,labels)
            loss.backward()
            optimizer.step()
        
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            torch.save(model.state_dict(), MODEL_PATH)
            best_accuracy = test_accuracy
