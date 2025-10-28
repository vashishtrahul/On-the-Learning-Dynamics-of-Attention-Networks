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


def create_mosaic_img(foreground_data,background_data,foreground_label,bg_idx,fg_idx,fg,m):
    """
    bg_idx : list of indexes of background data to be used as background image in mosaic
    fg_idx:
    fg:
    m:
    """
    fg1, fg2, fg3 = 0,1,2
    image_list = []
    j =0 
    for i in range(m):
        if i!=fg:
            image_list.append(background_data[bg_idx[j]])
            j+=1
        else:
            image_list.append(foreground_data[fg_idx])
            label = foreground_label[fg_idx]-fg1
    image_list = torch.stack(image_list)
    return image_list,label

