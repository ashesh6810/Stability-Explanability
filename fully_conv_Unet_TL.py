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
from saveNCfile_for_activations import savenc_for_activations
from data_loader import load_test_data
from data_loader import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage


### PATHS and FLAGS ###

path_static_activations = '/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/new_system_multistep_Unet/activations_analysis/'
path_weights = '/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/new_system_multistep_Unet/weights_analysis/'

FLAGS_WEIGHTS_DUMP=1
FLAGS_ACTIVATIONS_DUMP=1







##### prepare test data ###################################################

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/Dry5/PSI_output.nc')
lat=np.asarray(FF['lat'])
lon=np.asarray(FF['lon'])

lead = 1

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(FF,lead)
###############################################################################



################### Load training data files ########################################
fileList_train=[]
mylist = [1,2]
for k in mylist:
  fileList_train.append ('/glade/scratch/asheshc/theory-interp/QG/set'+str(k)+'/PSI_output.nc')
##########################################################################################

def store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training,epoch,out,x1,x2,x3,x4,x5,x6):

   Act_encoder[epoch,0,:,:,:,:] = x1.detach().cpu().numpy()
   Act_encoder[epoch,1,:,:,:,:] = x2.detach().cpu().numpy()
   Act_encoder[epoch,2,:,:,:,:] = x3.detach().cpu().numpy()
   Act_encoder[epoch,3,:,:,:,:] = x4.detach().cpu().numpy()

   Act_decoder1[epoch,:,:,:,:] = x5.detach().cpu().numpy()
   Act_decoder2[epoch,:,:,:,:] = x6.detach().cpu().numpy()





   output_training [epoch,:,:,:,:] = out.detach().cpu().numpy()

   return Act_encoder, Act_decoder1, Act_decoder2, output_training

def store_weights (net,epoch,hidden_weights_encoder,hidden_weights_decoder1,final_weights_network):

  hidden_weights_encoder[epoch,0,:,:,:,:] = net.hidden1.weight.data.cpu()
  hidden_weights_encoder[epoch,1,:,:,:,:] = net.hidden2.weight.data.cpu()
  hidden_weights_encoder[epoch,2,:,:,:,:] = net.hidden3.weight.data.cpu()
  hidden_weights_encoder[epoch,3,:,:,:,:] = net.hidden4.weight.data.cpu()


  hidden_weights_decoder1[epoch,:,:,:,:] = net.hidden5.weight.data.cpu()
  final_weights_network[epoch,:,:,:,:] = net.hidden6.weight.data.cpu()

  return hidden_weights_encoder, hidden_weights_decoder1, final_weights_network







class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = (nn.Conv2d(2, 64, kernel_size=5, stride=1, padding='same'))
        self.hidden1 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden2 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))


        self.hidden5 = (nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(192, 2, kernel_size=5, stride=1, padding='same' ))
    
    def forward (self,x):

        x1 = F.relu (self.input_layer(x))
        x2 = F.relu (self.hidden1(x1))
        x3 = F.relu (self.hidden2(x2))
        x4 = F.relu (self.hidden3(x3))

        x5 = torch.cat ((F.relu(self.hidden4(x4)),x3), dim =1)
        x6 = torch.cat ((F.relu(self.hidden5(x5)),x2), dim =1)
        

        out = (self.hidden6(x6))


        return out, x1, x2, x3, x4, x5, x6


net = CNN()

net.cuda()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print('**** Number of Trainable Parameters in BNN')
count_parameters(net)


batch_size = 100
num_epochs = 50
num_samples = 10
trainN = 7000

Act_encoder = np.zeros([num_epochs,4,num_samples,64,192,96])   #### Last three: number of channels, Nalt, Nlon
Act_decoder1 = np.zeros([num_epochs,num_samples,128,192,96])
Act_decoder2 = np.zeros([num_epochs,num_samples,192,192,96])
output_training = np.zeros([num_epochs,num_samples,2, 192, 96])

hidden_weights_encoder = np.zeros([num_epochs,4,64,64,5,5])
hidden_weights_decoder1 = np.zeros([num_epochs,128,128,5,5])
final_weights_network = np.zeros([num_epochs,2,192,5,5])




for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for loop in fileList_train:
     print('Training loop index',loop)

     psi_input_Tr_torch, psi_label_Tr_torch = load_train_data(loop, lead, trainN)

     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_input_Tr_torch[indices,:,:,:], psi_label_Tr_torch[indices,:,:,:]
        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output,_,_,_,_,_,_ = net(input_batch.cuda())
        loss = loss_fn(output, label_batch.cuda())
        loss.backward()
        optimizer.step()
        output_val,_,_,_,_,_,_ = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())
        val_loss = loss_fn(output_val, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())
        # print statistics

        if step % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, loss))
            print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, step + 1, val_loss))
            running_loss = 0.0
    out,x1,x2,x3,x4,x5,x6 = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())

    hidden_weights_encoder, hidden_weights_decoder1, final_weights_network = store_weights(net,epoch,hidden_weights_encoder, hidden_weights_decoder1, final_weights_network)

    Act_encoder, Act_decoder1, Act_decoder2, output_training = store_activations (Act_encoder,Act_decoder1,Act_decoder2,output_training, epoch,out,x1,x2,x3,x4,x5,x6)

print('Finished Training')


torch.save(net.state_dict(), './BNN_UNET_lead'+str(lead)+'.pt')

print('BNN Model Saved')

if (FLAGS_ACTIVATIONS_DUMP ==1):
 savenc_for_activations(Act_encoder, Act_decoder1, Act_decoder2,output_training,2,num_epochs,4,num_samples,64,128,192,192,96,path_static_activations+'BNN_UNET_Activations_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.nc')

 print('Saved Activations for BNN')

if (FLAGS_WEIGHTS_DUMP ==1):

 matfiledata = {}
 matfiledata[u'hidden_weights_encoder'] = hidden_weights_encoder
 matfiledata[u'hidden_weights_decoder'] = hidden_weights_decoder1
 matfiledata[u'final_layer_weights'] = final_weights_network
 hdf5storage.write(matfiledata, '.', path_weights+'BNN_UNET_Weights_Dry5_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.mat', matlab_compatible=True)

 print('Saved Weights for BNN')

############# Auto-regressive prediction #####################
M=1000
autoreg_pred = np.zeros([M,2,192,96])

for k in range(0,M):

  if (k==0):

    out,_,_,_,_,_,_ = (net(psi_test_input_Tr_torch[k].reshape([1,2,192,96]).cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

  else:

    out,_,_,_,_,_,_ = (net(torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,192,96])).float().cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

savenc(autoreg_pred, lon, lat, 'predicted_lead'+str(lead)+'.nc')
savenc(psi_test_label_Tr, lon, lat, 'truth_lead'+str(lead)+'.nc')
