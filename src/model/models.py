"""
This file contains the EncoderDecoderModel we used, we will soon upload the code to train these
model, currently we have only provided the model definations.
"""


import torch.nn as nn
import torch

class EncoderDecoderMLP(nn.Module):
    def __init__(self,input_dim,encode_dim):
        super(EncoderDecoderMLP,self).__init__()
        self.encode_dim = encode_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear((self.input_dim*self.input_dim*3),512,bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512,256,bias=True),
            nn.ReLU(True),
            nn.Linear(256,self.encode_dim,bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.encode_dim,256,bias=True),
            nn.ReLU(True),
            nn.Linear(256,512,bias=True),
            nn.ReLU(True),
            nn.Linear(512,(self.input_dim * self.input_dim *3), bias=True),
            )
        
    def forward(self,inp):
        encoding =  self.encoder(inp)
        decoding = self.decoder(encoding)
        return decoding
    def get_encoding(self,inp):
        return self.encoder(inp)

class EncoderDecoderConvNet(nn.Module):
    def __init__(self,channels):

        super(EncoderDecoderConvNet,self).__init__()
        self.encoder = nn.Sequential(
          nn.Conv2d(channels,32,4,2,1,bias=False),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(32,64,4,2,1,bias=False),
          nn.BatchNorm2d(64),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(64,128,4,2,1,bias=False),
          nn.BatchNorm2d(128),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(128,256,4,2,1,bias=False),
          nn.BatchNorm2d(256),
          nn.LeakyReLU(0.2,inplace=True),
          nn.Conv2d(256,512 , 4 , 1 , 0 ,bias=False))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512,256,4,1,0,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1,bias= False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32,channels,4,2,1,bias=False),
            nn.ReLU(True),
             )
    def forward(self,input):
        x = self.encoder(input)
        return self.decoder(x) , x
    def get_encoding(self,input):
        return self.encoder(input)

class Classifier(nn.Module):
    """
    """
    def __init__(self,num_classes):
        super(Classifier,self).__init__()
        self.fc1 = nn.Linear(640,256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256,num_classes)
        self.relu = nn.ReLU(True)
    def forward(self,input):
        x = self.relu(self.fc1(input))
        x = self.dropout(x)
        return self.fc2(x)