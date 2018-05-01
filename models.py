## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        conv1_channels = 32
        conv2_channels = conv1_channels * 2
        conv3_channels = conv2_channels * 2
        conv4_channels = conv3_channels * 2
        fc1_channels = 4096
        fc2_channels = 1028
        output_channels = 2*68
        weight_std = 0.005
        weight_mean = 0.0
        
        # 1 input image channel (grayscale), 64 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        # the output Tensor for one image, will have the dimensions: (64, 220, 220)
        self.conv1 = nn.Conv2d(1, conv1_channels, 5)
        self.conv1.weight.data.normal_(std=weight_std, mean=weight_mean)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # the output Tensor for one image, will have the dimensions: (64, 110, 110)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 64 input image channel, 128 output channels/feature maps
        # 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output Tensor will have the dimensions: (128, 106, 106)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 5)
        self.conv2.weight.data.normal_(std=weight_std, mean=weight_mean)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # the output Tensor for one image, will have the dimensions: (128, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 128 input image channel, 256 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 51
        # the output Tensor will have the dimensions: (256, 51, 51)
        self.conv3 = nn.Conv2d(conv2_channels, conv3_channels, 3)
        self.conv3.weight.data.normal_(std=weight_std, mean=weight_mean)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # the output Tensor will have the dimensions: (256, 25, 25)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 256 input image channel, 512 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (25-3)/1 +1 = 23
        # the output Tensor will have the dimensions: (512, 23, 23)
        self.conv4 = nn.Conv2d(conv3_channels, conv4_channels, 3)
        self.conv4.weight.data.normal_(std=weight_std, mean=weight_mean)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        # the output Tensor will have the dimensions: (512, 11, 11)
        self.pool4 = nn.MaxPool2d(2, 2)

        # 512 outputs * the 11*11 filtered/pooled map size
        self.fc1 = nn.Linear(conv4_channels*11*11, fc1_channels)
        self.fc1.weight.data.normal_(std=weight_std, mean=weight_mean)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        # 64 outputs * the 25*25 filtered/pooled map size
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.fc2.weight.data.normal_(std=weight_std, mean=weight_mean)
        
        # dropout with p=0.4
        self.fc2_drop = nn.Dropout(p=0.4)
        
        # finally, create 2*68 output channels (for the 68 keypoints)
        self.output = nn.Linear(fc2_channels, output_channels)
        self.output.weight.data.normal_(std=weight_std, mean=weight_mean)

        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # prep for linear layer
        x = x.view(x.size(0), -1)

        # 3 linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.output(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
