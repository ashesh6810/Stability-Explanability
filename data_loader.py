import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from saveNCfile import savenc


def load_test_data(FF,lead):

  psi=np.asarray(FF['z'])
 
  lat=np.asarray(FF['latitude'])
  lon=np.asarray(FF['longitude'])

  Nlat=np.size(lat,0)
  Nlon=np.size(lon,0);

  psi_test_input = psi[0:np.size(psi,0)-lead,:,:]
  psi_test_label = psi[0+lead:np.size(psi,0),:,:]


  psi_test_input = np.reshape(psi_test_input,(int(np.size(psi_test_input,0)),2,Nlat,Nlon)) 
  psi_test_label = np.reshape(psi_test_label,(int(np.size(psi_test_label,0)),2,Nlat,Nlon)) 



  psi_test_input_Tr=np.zeros([np.size(psi_test_input,0),2,Nlat,Nlon])
  psi_test_label_Tr=np.zeros([np.size(psi_test_label,0),2,Nlat,Nlon])



  for k in range(0,np.size(psi_test_input,0)):
    psi_test_input_Tr[k,0,:,:] = psi_test_input[k,0,:,:]
    psi_test_input_Tr[k,1,:,:] = psi_test_input[k,1,:,:]
    
    psi_test_label_Tr[k,0,:,:] = psi_test_label[k,0,:,:]
    psi_test_label_Tr[k,1,:,:] = psi_test_label[k,1,:,:]

## convert to torch tensor
  psi_test_input_Tr_torch = torch.from_numpy(psi_test_input_Tr).float()
  psi_test_label_Tr_torch = torch.from_numpy(psi_test_label_Tr).float()

  return psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr


def load_train_data(loop,lead,trainN):
  
     File=nc.Dataset(loop)

     psi=np.asarray(File['z'])
     
     lat=np.asarray(File['latitude'])
     lon=np.asarray(File['longitude'])

     Nlat=np.size(lat,0)
     Nlon=np.size(lon,0);    


     psi_train_input = psi[0:trainN,:,:]
     psi_train_label = psi[0+lead:trainN+lead,:,:]
     psi_train_input = np.reshape(psi_train_input,(int(np.size(psi_train_input,0)),2,Nlat,Nlon))
     psi_train_label = np.reshape(psi_train_label,(int(np.size(psi_train_label,0)),2,Nlat,Nlon))



     psi_train_input_Tr=np.zeros([np.size(psi_train_input,0),2,Nlat,Nlon])
     psi_train_label_Tr=np.zeros([np.size(psi_train_label,0),2,Nlat,Nlon])

     for k in range(0,trainN):
      psi_train_input_Tr[k,0,:,:] = psi_train_input[k,0,:,:]
      psi_train_input_Tr[k,1,:,:] = psi_train_input[k,1,:,:]
      psi_train_label_Tr[k,0,:,:] = psi_train_label[k,0,:,:]
      psi_train_label_Tr[k,1,:,:] = psi_train_label[k,1,:,:]
    

     print('Train input', np.shape(psi_train_input_Tr))
     print('Train label', np.shape(psi_train_label_Tr))
     psi_train_input_Tr_torch = torch.from_numpy(psi_train_input_Tr).float()
     psi_train_label_Tr_torch = torch.from_numpy(psi_train_label_Tr).float()  

     return psi_train_input_Tr_torch, psi_train_label_Tr_torch
