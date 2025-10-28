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





def calculate_loss_focus(gamma,focus_output):
    
    if len(focus_output.shape)>2:
        focus_output = focus_output[:,:,0]
 
    
    log_outputs = torch.log_softmax(focus_output,dim=1)
    loss_ = gamma*log_outputs
    loss_ = torch.sum(loss_,dim=1)
    loss_ = -torch.mean(loss_,dim=0)    
    return loss_ 
    




def calculate_loss_classification(gamma,classification_output,label,criterion,n_patches):
    batch = label.size(0)
    n_classes = classification_output.size(2)
    label = label.repeat_interleave(n_patches)
    classification_output = classification_output.reshape((batch*n_patches,n_classes))
    loss_ = criterion(classification_output,label)
    
    loss_ = loss_.reshape((batch,n_patches))
    
    loss_ = gamma*loss_
    loss_ = torch.sum(loss_,dim=1)
    loss_ = torch.mean(loss_,dim=0)
    
    return loss_



def expectation_step(fc,cl,data,labels):
    batch= data.size(0)
    patches = data.size(1)
    dims = data.size(2)
    with torch.no_grad():
        outputs_f = torch.softmax(fc(data),dim=1)
        if len(data.shape)>3:
            data = data.reshape(batch*patches,3,32,32)
        else:
            dims = data.size(2)
            data = data.reshape(batch*patches,dims)
        outputs_g = cl(data)
        n_classes = outputs_g.size(1)
        
    outputs_g = torch.softmax(outputs_g.reshape(batch,patches,n_classes),dim=2)

    outputs_g = outputs_g[np.arange(batch),:,labels]
    
    
    if len(outputs_f.shape)>2:
        outputs_f = outputs_f[:,:,0]  
    p_x_y_z = outputs_f*outputs_g   #(1-outputs_g)    
    
    
    normalized_p = p_x_y_z/torch.sum(p_x_y_z,dim=1,keepdims=True)
#     print(outputs_f[0],outputs_g[0],normalized_p[0])
    return normalized_p


def maximization_step(p_z,focus,classification,data,labels,focus_optimizer,classification_optimizer,Criterion):    
    batch = data.size(0)
    patches = data.size(1)
    focus_optimizer.zero_grad()
    classification_optimizer.zero_grad()
    
    focus_outputs = focus(data)
    if len(data.shape)>3:
        data = data.reshape(batch*patches,3,32,32)
    else:
        dims = data.size(2)
        data = data.reshape(batch*patches,dims)
        
    classification_outputs = classification(data) 
    n_classes = classification_outputs.size(1)
    classification_outputs = classification_outputs.reshape(batch,patches,n_classes)
    
    
    
    loss_focus = calculate_loss_focus(p_z,focus_outputs)
    loss_classification = calculate_loss_classification(p_z,classification_outputs,
                                                        labels,Criterion,patches)
    
  
    loss_focus.backward() 
    loss_classification.backward()
    focus_optimizer.step()
    classification_optimizer.step()
    
    return focus,classification,focus_optimizer,classification_optimizer



# method 1
def evaluation_method_1(dataloader,focus,classification):
    """
    returns \sigma_k(g(x_j*)) j* is the argmax_j(\sigma_j(XU))
    """
    predicted_indexes = []
    foreground_index_list = []
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for j,data in enumerate(dataloader):
            images,labels,foreground_index = data
            images = images.float()
            images = images.to(device)
            foreground_index_list.append(foreground_index.numpy())
            labels_list.append(labels.numpy())
            batch = images.size(0)
            scores = focus(images)
            if len(scores.shape)>2:
                indexes = torch.argmax(F.softmax(scores,dim=1),dim=1).cpu().numpy()[:,0]
            else:
                indexes = torch.argmax(F.softmax(scores,dim=1),dim=1).cpu().numpy()
            predicted_indexes.append(indexes)
            outputs = F.softmax(classification(images[np.arange(batch),indexes,:]),dim=1)
            prediction = torch.argmax(outputs,dim=1)
            prediction_list.append(prediction.cpu().numpy())

    predicted_indexes = np.concatenate(predicted_indexes,axis=0)
    foreground_index_list = np.concatenate(foreground_index_list,axis=0)
    prediction_list = np.concatenate(prediction_list,axis=0)
    labels_list = np.concatenate(labels_list,axis=0)
    
    #print(predicted_indexes.shape,foreground_index_list.shape)

    ftpt = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpt  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ftpf  = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpf  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
#     focus_true = (np.sum(predicted_indexes == foreground_index_list,axis=0).item()/
#                         len(foreground_index_list))*100
    accuracy = (np.sum(prediction_list == labels_list,axis=0)/len(labels_list) )*100
    
    return ftpt,ffpt,ftpf,ffpf,accuracy





def my_cross_entropy(output,target,alpha,criterion):
    
    batch = alpha.size(0)
    patches = alpha.size(1)
    target = target.repeat_interleave(patches)
    loss = criterion(output,target)
    loss = torch.reshape(loss,(batch,patches))
    if len(alpha.shape)>2:
        final_loss = torch.sum(loss*alpha[:,:,0],dim=1)
    else:
        final_loss = torch.sum(loss*alpha,dim=1)
    final_loss = torch.mean(final_loss,dim=0)
    return final_loss



def marginal_loss(fc,cl,data,labels):
    batch = data.size(0)
    patches = data.size(1)
    outputs_f = torch.softmax(fc(data),dim=1)
    if len(outputs_f.shape)>2:
        outputs_f = outputs_f[:,:,0]
    if len(data.shape)>3:
        data = data.reshape(batch*patches,3,32,32)
    else:
        dims = data.size(2)
        data = data.reshape(batch*patches,dims)
    outputs_g = cl(data)
    n_classes = outputs_g.size(1)
    outputs_g = torch.softmax(outputs_g.reshape(batch,patches,n_classes),dim=2)
    outputs_g = outputs_g[np.arange(batch),:,labels]
    #print(outputs_f.shape,outputs_g.shape)
    p_x_y_z  = outputs_f*outputs_g
    #print("Flag 1", torch.sum(p_x_y_z,dim=1,keepdims=True) )
    loss  = -torch.mean(torch.log(torch.sum(p_x_y_z,dim=1,keepdims=True)+1e-30))
    return loss




def evaluation_method_2(dataloader,focus,classification):
    """
    returns \sum_j(\alpha_j * \sigma_k(g(x_j)) 
    """
    predicted_indexes = []
    foreground_index_list = []
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for j,data in enumerate(dataloader):
            images,labels,foreground_index = data
            images = images.float()
            images = images.to(device)
            batch = images.size(0)
            patches = images.size(1)
            foreground_index_list.append(foreground_index.numpy())
            labels_list.append(labels.numpy())
            batch = images.size(0)
            focus_outputs = F.softmax(focus(images),dim=1)
            if len(focus_outputs.shape)>2:
                focus_outputs = focus_outputs[:,:,0]
            indexes = torch.argmax(focus_outputs,dim=1).cpu().numpy()
            predicted_indexes.append(indexes)
            
            if len(images.shape)>3:
                images = images.reshape(batch*patches,3,32,32)
            else:
                dims = images.size(2)
                images = images.reshape(batch*patches,dims)
            classification_outputs = F.softmax(classification(images),dim=1)
            n_classes = classification_outputs.size(1)
            classification_outputs = classification_outputs.reshape(batch,patches,n_classes)

            #print(classification_outputs.shape,focus_outputs.shape)
            if len(images.shape)>3:
                focus_outputs = focus_outputs[:,:,None]
            else:
                focus_outputs = focus_outputs[:,:,None]
            prediction = torch.argmax(torch.sum(focus_outputs*classification_outputs,dim=1),dim=1)
            
           
            prediction_list.append(prediction.cpu().numpy())

    predicted_indexes = np.concatenate(predicted_indexes,axis=0)
    foreground_index_list = np.concatenate(foreground_index_list,axis=0)
    prediction_list = np.concatenate(prediction_list,axis=0)
    labels_list = np.concatenate(labels_list,axis=0)
    print(prediction_list.shape)

    ftpt = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpt  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ftpf  = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpf  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
#     focus_true = (np.sum(predicted_indexes == foreground_index_list,axis=0).item()/
#                         len(foreground_index_list))*100
    accuracy = (np.sum(prediction_list == labels_list,axis=0)/len(labels_list) )*100
    
    return ftpt,ffpt,ftpf,ffpf,accuracy



# method 3
def evaluation_method_3(dataloader,focus,classification):
    """
    returns \sum_j( \sigma_k(\alpha_j * g(x_j)) 
    """
    
    predicted_indexes = []
    foreground_index_list = []
    prediction_list = []
    labels_list = []
    with torch.no_grad():
        for j,data in enumerate(dataloader):
            images,labels,foreground_index = data
            images = images.float()
            images = images.to(device)
            foreground_index_list.append(foreground_index.numpy())
            labels_list.append(labels.numpy())
            batch = images.size(0)
            scores = focus(images)
            alphas = F.softmax(scores,dim=1)
            if len(scores.shape)>2:
                indexes = torch.argmax(F.softmax(scores,dim=1),dim=1).cpu().numpy()[:,0]
            else:
                indexes = torch.argmax(F.softmax(scores,dim=1),dim=1).cpu().numpy()
            predicted_indexes.append(indexes)
            if len(images.shape)>3:
                images = torch.sum(alphas[:,:,None,None,None]*images,dim=1)
            else:
                images = torch.sum(alphas*images,dim=1)
            
            outputs = F.softmax(classification(images),dim=1)
            prediction = torch.argmax(outputs,dim=1)
            prediction_list.append(prediction.cpu().numpy())
#     print(len(predicted_indexes))
    predicted_indexes = np.concatenate(predicted_indexes,axis=0)
    foreground_index_list = np.concatenate(foreground_index_list,axis=0)
    prediction_list = np.concatenate(prediction_list,axis=0)
    labels_list = np.concatenate(labels_list,axis=0)


    ftpt = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpt  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list == labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ftpf  = (np.sum(np.logical_and(predicted_indexes == foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
    
    ffpf  = (np.sum(np.logical_and(predicted_indexes != foreground_index_list,
                                 prediction_list != labels_list),axis=0).item()/len(foreground_index_list))*100
#     focus_true = (np.sum(predicted_indexes == foreground_index_list,axis=0).item()/
#                         len(foreground_index_list))*100
    accuracy = (np.sum(prediction_list == labels_list,axis=0)/len(labels_list) )*100
    
    return ftpt,ffpt,ftpf,ffpf,accuracy


def train_model_sin(data,alpha,classification,classification_optimizer,Criterion):
    
    
    images,labels,fore_idx = data
    batch = images.size(0)
    patches = images.size(1)
    images= images.float()
    images,labels = images.to(device),labels.to(device)
            
 
    classification_optimizer.zero_grad()    
    alpha_weights = torch.ones((batch,patches),device=device)*((1-alpha)/(patches-1))
    alpha_weights[:,0] = torch.ones(batch,device=device)*alpha
    
    if len(images.shape)>3:
        images =  images.reshape(batch*patches,3,32,32)
    else:
        dims = images.size(2)
        images = images.reshape(batch*patches,dims)
    outputs = classification(images)
    loss = my_cross_entropy(outputs,labels,alpha_weights,Criterion)   
    loss.backward()
    classification_optimizer.step()
        
    return classification,classification_optimizer


def train_model_cin(data,alpha,classification,classification_optimizer,Criterion):
    
    
    images,labels,fore_idx = data
    batch = images.size(0)
    patches = images.size(1)
    images = images.float()
    images,labels = images.to(device),labels.to(device)
            
 

    classification_optimizer.zero_grad() 
    
    if len(images.shape)>3:
        alpha_weights = torch.ones((batch,patches,3,32,32),device=device)*((1-alpha)/(patches-1))
        alpha_weights[:,0] = torch.ones((batch,3,32,32),device=device)*alpha
    else:
        dims = images.size(2)
        alpha_weights = torch.ones((batch,patches,dims),device=device)*((1-alpha)/(patches-1))
        alpha_weights[:,0] = torch.ones((batch,dims),device=device)*alpha
        
        
   

    images = torch.mul(alpha_weights,images)
    
    #print(images.shape)

    
    outputs = classification(torch.sum(images,dim=1))
    loss = Criterion(outputs,labels)   
    loss.backward()
    classification_optimizer.step()
        
    return classification,classification_optimizer



def train_model(data,focus,classification,focus_optimizer,classification_optimizer,Criterion):
    images,labels,fore_idx = data
    images = images.float()
    batch = images.size(0)
    patches = images.size(1)
    images = images.float()
    images,labels = images.to(device),labels.to(device)
            
    focus_optimizer.zero_grad()
    classification_optimizer.zero_grad()
            
    alphas = torch.softmax(focus(images),dim=1)
    if len(images.shape)>3:
        images =  images.reshape(batch*patches,3,32,32)
    else:
        dims = images.size(2)
        images = images.reshape(batch*patches,dims)
    outputs = classification(images)
    loss = my_cross_entropy(outputs,labels,alphas,Criterion)
            
    loss.backward()
    focus_optimizer.step()
    classification_optimizer.step()
        
    return focus,classification,focus_optimizer,classification_optimizer

def train_model_sa(data,focus,classification,focus_optimizer,classification_optimizer,Criterion):
    images,labels,fore_idx = data
    batch = images.size(0)
    patches = images.size(1)
    images= images.float()
    images,labels = images.to(device),labels.to(device)
            
    focus_optimizer.zero_grad()
    classification_optimizer.zero_grad()
    alphas = torch.softmax(focus(images),dim=1)
    
    #print(alphas)
    if len(images.shape)>3:
        images = torch.sum(alphas[:,:,None,None,None]*images,dim=1)
    else:
        images = torch.sum(alphas*images,dim=1)
    outputs = classification(images)
    loss = Criterion(outputs,labels)   
    loss.backward()
    focus_optimizer.step()
    classification_optimizer.step()
        
    return focus,classification,focus_optimizer,classification_optimizer


def marginal_loss_sin(cl,data,labels,alpha):
    batch = data.size(0)
    patches = data.size(1)
    
    
    alpha_weights = torch.ones((batch,patches),device=device)*((1-alpha)/(patches-1))
    alpha_weights[:,0] = torch.ones(batch,device=device)*alpha
    
    
    outputs_f = alpha_weights
    
    if len(data.shape)>3:
        data = data.reshape(batch*patches,3,32,32)
    else:
        dims = data.size(2)
        data = data.reshape(batch*patches,dims)
    outputs_g = cl(data)
    
    n_classes = outputs_g.size(1)
    outputs_g = torch.softmax(outputs_g.reshape(batch,patches,n_classes),dim=2)
    outputs_g = outputs_g[np.arange(batch),:,labels]
    

    
    p_x_y_z  = outputs_f*outputs_g
    loss  = -torch.mean(torch.log(torch.sum(p_x_y_z,dim=1,keepdims=True)))
    
    return loss


