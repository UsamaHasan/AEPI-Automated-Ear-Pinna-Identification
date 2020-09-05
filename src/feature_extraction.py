import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader , TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  sklearn.metrics import mutual_info_score
from utils.utils import matplotlib_imshow

def distance_measure(matrix1 , matrix2):
    """
    Distance formula to compare squared distance between feature vectors.
    """
    difference = np.square(matrix2 - matrix1)
    difference = np.sqrt(difference)
    return (np.sum(difference))/len(matrix1)

if __name__ == '__main__':

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
    y_test =  np.array(y_test)
    y_train = y_train - 1
    y_test = y_test - 1

    index_sample_1 = np.where(y_train == 48)
    index_sample_2 = np.where(y_train == 156)
    sample_1 = x_train[index_sample_1]
    sample_2 = x_train[index_sample_2]
    
    sample_1 = torch.Tensor(sample_1)
    sample_2 = torch.Tensor(sample_2)
    # Initializing model
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = models.resnet101(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    #model =  models.vgg16(pretrained=True)
    #feature_extractor = torch.nn.Sequential( model.features)
    total_cross_similarity = 0.0
    total_mutual_similarity = 0.0
    total_cross_distance = 0.0
    total_same_class_distance = 0.0 
    i = 0
    while( i < (len(sample_1) -1) and i < (len(sample_2) - 1)):    
        c_11 = sample_1[i].view(1,sample_1[i].size(0),sample_1[i].size(1),sample_1[i].size(2))
        c_12 = sample_1[i+1].view(1,sample_1[i+1].size(0),sample_1[i+1].size(1),sample_1[i+1].size(2))
        c_21 = sample_2[i].view(1,sample_2[i].size(0),sample_2[i].size(1),sample_2[i].size(2))
        c_22 = sample_2[i+1].view(1,sample_2[i+1].size(0),sample_2[i+1].size(1),sample_2[i+1].size(2))
        
        output_11 = feature_extractor(c_11)
        output_12 = feature_extractor(c_12)
        output_21 = feature_extractor(c_21)
        output_22 = feature_extractor(c_22)

        sample_11 = output_11.detach().numpy()
        sample_12 = output_12.detach().numpy()
        sample_21 = output_21.detach().numpy()
        sample_22 = output_22.detach().numpy()
        
        sample_11 = sample_11.reshape(sample_11.shape[1])
        sample_12=  sample_12.reshape(sample_12.shape[1])
        sample_21 = sample_21.reshape(sample_21.shape[1])
        sample_22=  sample_22.reshape(sample_22.shape[1])

        cross_distance = distance_measure(sample_11,sample_22)
        match_distance = distance_measure(sample_11,sample_12)

        cross_similarity = mutual_info_score(sample_11,sample_22)
        mutual_similarity   = mutual_info_score(sample_11,sample_12)
        total_cross_similarity = cross_similarity + total_cross_similarity
        total_mutual_similarity = total_mutual_similarity + mutual_similarity
        total_cross_distance = total_cross_distance + cross_distance
        total_same_class_distance = total_same_class_distance + match_distance
        
        print(f'L1 Norm Cross Sample: {cross_distance}')
        print(f'L1 Norm: {match_distance}')
        print(f'Cross Similarity :{ cross_similarity}')
        print(f'Mutual Similarity :{ mutual_similarity}')
        i+=1
    print(f'Total Cross Similarity:{total_cross_similarity/i}')
    print(f'Total Mutual Similarity:{total_mutual_similarity/i}')
    print(f'Total Cross Distance:{total_cross_distance/i}')
    print(f'Total Mutual Class Distance:{total_same_class_distance/i}')
