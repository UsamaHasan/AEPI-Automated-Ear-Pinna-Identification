import torch.nn as nn
import torch
from torchvision import models 
import torch.nn.functional as F

class LMCL(nn.Module):
    def __init__(self, embedding_size, num_classes, s, m):
        super(LMCL, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weights = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.kaiming_normal_(self.weights)

    def forward(self, embedding, label):
        assert embedding.size(1) == self.embedding_size, 'embedding size wrong'
        logits = F.linear(F.normalize(embedding), F.normalize(self.weights))
        margin = torch.zeros_like(logits)
        margin.scatter_(1, label.view(-1, 1), self.m)
        m_logits = self.s * (logits - margin)
        return logits, m_logits, self.s * F.normalize(embedding), F.normalize(self.weights)

class Spnet(nn.Module):
    def __init__(self,NUM_CLASSES,state_dict=None):
        super(Spnet, self).__init__()
        self.NUM_CLASSES = NUM_CLASSES
        
        model = models.resnet152(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
        self.linear1 = nn.Linear(2048+512,2048)
        self.bn1 = nn.BatchNorm1d(num_features=2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(num_features=1024)
        self.linear3 = nn.Linear(1024, self.NUM_CLASSES)
        self.binary_layer = nn.Linear(1024,1)       
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(True)
        self.block = self.spatial_encoder()
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self.loss_lmcl = LMCL(embedding_size=1024, num_classes=self.NUM_CLASSES, s = 8, m=0.2)
    
    def forward(self,x,labels=None):
        # Extract Features through ResNet Backbone feature extractor
        features = self.feature_extractor(x)
        #Extract Spatial Features 
        latent = self.block(x)
        features = torch.flatten(features,1)        
        latent = torch.flatten(latent,1)
        #Concat Both the tensors
        x = torch.cat((features,latent),1)
        #Linear Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #Dropout layer
        x = self.dropout(x)
        #Linear layer 2
        x = self.linear2(x)
        x = self.relu(self.bn2(x))
        #class_l = self.linear3(x)
        gender_l = self.sigmoid(self.binary_layer(x))
        if labels is not None:
            class_l = self.loss_lmcl(x,labels)
        else:
            class_l = self.linear3(x)
        return class_l , gender_l

    def spatial_encoder(self):
        
        return torch.nn.Sequential(torch.nn.Conv2d(3,64,(1,3),stride=2,groups=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(64,128,(3,1),stride=2,groups=64),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(128,64,(1,1),stride=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(64,128,(1,2),stride=2,groups=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(128,256,(2,1),stride=2,groups=128),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(256,128,(1,1),stride=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(128,256,(1,2),stride=2,groups=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  torch.nn.Conv2d(256,512,(2,1),stride=2,groups=256))