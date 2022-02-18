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

  psi=FF['PSI']
  psi=psi[2500:,:,:,:]
  lat=np.asarray(FF['lat'])
  lon=np.asarray(FF['lon'])

  Nlat=np.size(psi,2);
  Nlon=np.size(psi,3);

  psi_test_input = psi[0:np.size(psi,0)-lead,:,:,:]
  psi_test_label = psi[0+lead:np.size(psi,0),:,:,:]



  psi_test_input_Tr=np.zeros([np.size(psi,0),2,Nlat,Nlon])
  psi_test_label_Tr=np.zeros([np.size(psi,0),2,Nlat,Nlon])



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

     psi=File['PSI']
     psi=psi[2500:,:,:,:]
     Nlat=np.size(psi,2);
     Nlon=np.size(psi,3);
     


     psi_input = psi[0:trainN,:,:,:]
     psi_label = psi[0+lead:trainN+lead,:,:,:]

     psi_input_Tr=np.zeros([trainN,2,Nlat,Nlon])
     psi_label_Tr=np.zeros([trainN,2,Nlat,Nlon])


     for k in range(0,trainN):
      psi_input_Tr[k,0,:,:] = psi_input[k,0,:,:]
      psi_input_Tr[k,1,:,:] = psi_input[k,1,:,:]
      psi_label_Tr[k,0,:,:] = psi_label[k,0,:,:]
      psi_label_Tr[k,1,:,:] = psi_label[k,1,:,:]

     print('Train input', np.shape(psi_input_Tr))
     print('Train label', np.shape(psi_label_Tr))
     psi_input_Tr_torch = torch.from_numpy(psi_input_Tr).float()
     psi_label_Tr_torch = torch.from_numpy(psi_label_Tr).float()  

     return psi_input_Tr_torch, psi_label_Tr_torch
