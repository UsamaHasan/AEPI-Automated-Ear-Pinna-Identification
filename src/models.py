import torch.nn as nn
import torch

class Net(nn.Module):
    """"""
    def __init__(self):
        super(Net,self).__init__()
        self.NUM_CLASSES = 164
        self.levels = [4,2,1]
        model = models.resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(*(list(model.children())[:-1]))
        self.fc1 = nn.Linear(2560,1024,bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc3 = nn.Linear(1024,self.NUM_CLASSES)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU(True)
        self.block = self.separable_conv_block()
    
    def forward(self,x):
        features = self.feature_extractor(x)
        edges = self.block(x)
        features = torch.flatten(features,1)        
        edges = torch.flatten(edges,1)
        x = torch.cat((features,edges),1)
        x = self.fc1(x)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
    def spatial_encoder_block(self):
        return torch.nn.Sequential( nn.Conv2d(3,64,(1,3),stride=3,groups=1),
                             nn.ReLU(),
                             nn.Conv2d(64,128,(3,1),stride=3,groups=64),
                             nn.ReLU(),
                             nn.Conv2d(128,256,(1,3),stride=3,groups=1),
                             nn.ReLU(),
                             nn.Conv2d(256,512,(3,1),stride=3,groups=256),
                             nn.ReLU())
    
    def spatial_pyramid_pool(self,previous_conv, previous_conv_size, mode ):
        """
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(self.levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / self.levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / self.levels[i]))
            w_pad1 = int(math.floor((w_kernel * self.levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * self.levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * self.levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * self.levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * self.levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * self.levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

class EncoderDecoderMLP(nn.Module):
    """
    AutoEncoder.
    """
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
        """
        """
        return self.encoder(inp)

class EncoderDecoderConvNet(nn.Module):
    """
    Convolutional AutoEncoder.
    """
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
        return self.decoder(x)
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
