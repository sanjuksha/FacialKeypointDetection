## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=3 )# Output size = (224-3/3  +1) =75
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2) # o/p =75-2/2  + 1 = 38
        #self.conv1_drop = nn.Dropout(p=0.3)
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(32, 64, 3, stride=3) # o/p 38-3/3 +1 = 13
        self.pool = nn.MaxPool2d(3,3)# o/p 13-3/3 + 1 = 5
        #self.conv2_drop = nn.Dropout(p=0.4)
        self.conv3 = nn.Conv2d(64,128, 2, stride=1) # o/p 5-2/1 +1 = 4
        self.pool = nn.MaxPool2d(2,2)# o/p 4-2/1 + 1 = 3
        self.conv3_drop = nn.Dropout(p=0.4)
        
        # 20 outputs * the 5*5 filtered/pooled map size
        # 10 output channels (for the 10 classes)
        #self.conv3 = nn.Conv2d(64, 128, 3, stride=3) # o/p 54-3/3 +1 = 18
        #self.pool = nn.MaxPool2d(2,2)# o/p 18-2/2 + 1 = 9
        self.fc1 = nn.Linear(512, 228)
        self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(228,136)
        #self.fc2_drop = nn.Dropout(p=0.4)
        
#         self.fc2_drop = nn.Dropout(p=0.4)
#         self.fc3 = nn.Linear(228,136)
  
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.elu(self.conv1(x)))
        #x = self.conv1_drop(x)
        x = self.pool(F.elu(self.conv2(x)))
        #x = self.conv2_drop(x)
        x = self.pool(F.elu(self.conv3(x)))
        x = self.conv3_drop(x)
#         x = self.pool(F.tanh(self.conv3(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # one linear layer
        x = F.elu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.elu(self.fc2(x))
        #x = self.fc2_drop(x)

        
        # final output
        return x
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
