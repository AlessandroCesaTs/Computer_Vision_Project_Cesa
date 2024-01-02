import torch
from torch import nn

#build the network

class simpleCNN(nn.Module):
  def __init__(self):
    super(simpleCNN,self).__init__() #initialize the model

    self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,stride=1) #Output image size is (size+2*padding-kernel)/stride -->62*62
    self.relu1=nn.ReLU()
    self.maxpool1=nn.MaxPool2d(kernel_size=2,stride=2) #outtput image 62/2-->31*31

    self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=1) #output image is 29*29
    self.relu2=nn.ReLU()
    self.maxpool2=nn.MaxPool2d(kernel_size=2,stride=2) #output image is 29/2-->14*14  (MaxPool2d approximates size with floor)

    self.conv3=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1) #output image is 12*12
    self.relu3=nn.ReLU()

    self.fc1=nn.Linear(32*12*12,15) #16 channels * 16*16 image (64*64 with 2 maxpooling of stride 2), 15 output features=15 classes

  def forward(self,x):
    x=self.conv1(x)
    x=self.relu1(x)
    x=self.maxpool1(x)

    x=self.conv2(x)
    x=self.relu2(x)
    x=self.maxpool2(x)

    x=self.conv3(x)
    x=self.relu3(x)

    x=x.view(-1,32*12*12)

    x=self.fc1(x)

    return x