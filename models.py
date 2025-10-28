#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision

import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm as tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[ ]:


class Focus_linear(nn.Module):
    def __init__(self,input_dims):
        super(Focus_linear,self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(input_dims,1,bias=False)
    def forward(self,x):
        x = self.fc1(x)
        return x
      


class Focus_non_linear(nn.Module):
    def __init__(self,input_dims):
        super(Focus_non_linear,self).__init__()
        self.input_dims = input_dims
        self.fc1 = nn.Linear(input_dims,50,bias=False)
        self.fc2 = nn.Linear(50,1,bias=False)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class Classification_linear(nn.Module):
    def __init__(self,input_dims,output_dims):
        super(Classification_linear,self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.fc1 = nn.Linear(input_dims,output_dims)
    def forward(self,x):
        x = self.fc1(x)
        return x
    
    
    
def initialize_weights_zero(m):
    if isinstance(m,nn.Linear):
        torch.nn.init.zeros_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# In[ ]:


class Focus_cnn(nn.Module):
  def __init__(self):
    super(Focus_cnn, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, bias=False)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, bias=False)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, bias=False)
    self.fc1 = nn.Linear(1024, 512, bias=False)
    self.fc2 = nn.Linear(512, 64, bias=False)
    self.fc3 = nn.Linear(64, 10, bias=False)
    self.fc4 = nn.Linear(10,1, bias=False)

    torch.nn.init.xavier_normal_(self.conv1.weight)
    torch.nn.init.xavier_normal_(self.conv2.weight)
    torch.nn.init.xavier_normal_(self.conv3.weight)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)
    torch.nn.init.xavier_normal_(self.fc3.weight)
    torch.nn.init.xavier_normal_(self.fc4.weight)

  def forward(self,z):  #y is avg image #z batch of list of 9 images
    batch = z.size(0)
    patches = z.size(1)
    z = z.view(batch*patches,3,32,32)
    alpha =  self.helper(z)
    alpha = alpha.view(batch,patches,-1)
    return alpha[:,:,0] # scores 
    
  def helper(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    # print(x.shape)
    x = (F.relu(self.conv3(x)))
    x =  x.view(x.size(0), -1)
    # print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x


class Classification_cnn(nn.Module):
  def __init__(self,output_dims):
    super(Classification_cnn, self).__init__()
    self.output_dims = output_dims
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)
    self.fc4 = nn.Linear(10,self.output_dims)

    torch.nn.init.xavier_normal_(self.conv1.weight)
    torch.nn.init.zeros_(self.conv1.bias)
    torch.nn.init.xavier_normal_(self.conv2.weight)
    torch.nn.init.zeros_(self.conv2.bias)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.zeros_(self.fc1.bias)
    torch.nn.init.xavier_normal_(self.fc2.weight)
    torch.nn.init.zeros_(self.fc2.bias)
    torch.nn.init.xavier_normal_(self.fc3.weight)
    torch.nn.init.zeros_(self.fc3.bias)
    torch.nn.init.xavier_normal_(self.fc4.weight)
    torch.nn.init.zeros_(self.fc4.bias)

  def forward(self,z): 
    y1 = self.pool(F.relu(self.conv1(z)))
    y1 = self.pool(F.relu(self.conv2(y1)))
    y1 = y1.view(-1, 16 * 5 * 5)

    y1 = F.relu(self.fc1(y1))
    y1 = F.relu(self.fc2(y1))
    y1 = F.relu(self.fc3(y1))
    y1 = self.fc4(y1)
    return y1 


class Classification_cnn1(nn.Module):
  def __init__(self,output_dims):
    super(Classification_cnn1, self).__init__()
    self.output_dims = output_dims
    self.conv1 = nn.Conv2d(3, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(16, 32, 5)
    self.fc1 = nn.Linear(32 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, self.output_dims)

    torch.nn.init.xavier_normal_(self.conv1.weight)
    torch.nn.init.zeros_(self.conv1.bias)
    torch.nn.init.xavier_normal_(self.conv2.weight)
    torch.nn.init.zeros_(self.conv2.bias)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.zeros_(self.fc1.bias)
    torch.nn.init.xavier_normal_(self.fc2.weight)
    torch.nn.init.zeros_(self.fc2.bias)

  def forward(self,z): 
    y1 = self.pool(F.relu(self.conv1(z)))
    y1 = self.pool(F.relu(self.conv2(y1)))
    y1 = y1.view(-1, 32 * 5 * 5)

    y1 = F.relu(self.fc1(y1))
    y1 = self.fc2(y1)
    return y1 


class Focus_cnn1(nn.Module):
  def __init__(self):
    super(Focus_cnn1, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0, bias=False)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, bias=False)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, bias=False)
    self.fc1 = nn.Linear(1024, 512, bias=False)
    self.fc2 = nn.Linear(512, 64, bias=False)
    self.fc3 = nn.Linear(64, 1, bias=False)

    torch.nn.init.xavier_normal_(self.conv1.weight)
    torch.nn.init.xavier_normal_(self.conv2.weight)
    torch.nn.init.xavier_normal_(self.conv3.weight)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)
    torch.nn.init.xavier_normal_(self.fc3.weight)

  def forward(self,z):  #y is avg image #z batch of list of 9 images
    batch = z.size(0)
    patches = z.size(1)
    z = z.view(batch*patches,3,32,32)
    alpha =  self.helper(z)
    alpha = alpha.view(batch,patches,-1)
    return alpha[:,:,0] # scores 
    
  def helper(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    # print(x.shape)
    x = (F.relu(self.conv3(x)))
    x =  x.view(x.size(0), -1)
    # print(x.shape)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x