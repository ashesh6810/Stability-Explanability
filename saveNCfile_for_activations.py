import numpy as np
import netCDF4 as nc4

def savenc_for_activations(x1,x2,x3,y,num_level,num_epochs,num_layers,num_samples,num_filters1,num_filters2,num_filters3,lon,lat,filename):
    f = nc4.Dataset(filename,'w', format='NETCDF4')
    tempgrp = f.createGroup('Temp_data')
    tempgrp.createDimension('lon', lon)
    tempgrp.createDimension('lat', lat)
    tempgrp.createDimension('num_filters1',num_filters1)
    tempgrp.createDimension('num_filters2',num_filters2)
    tempgrp.createDimension('num_filters3',num_filters3)
    tempgrp.createDimension('samples', num_samples)
    tempgrp.createDimension('layers', num_layers)
    tempgrp.createDimension('epochs', num_epochs)
    tempgrp.createDimension('level', )

    
  
    output = tempgrp.createVariable('Output', 'f4',('epochs','samples','level','lon','lat') )  
    psi1 = tempgrp.createVariable('Activation_encoder', 'f4', ('epochs','layers','samples','num_filters1','lon','lat'))
    psi2 = tempgrp.createVariable('Activation_decoder1', 'f4',('epochs','samples','num_filters2','lon','lat') )
    psi3 = tempgrp.createVariable('Activation_decoder2', 'f4',('epochs','samples','num_filters3','lon','lat') )
    
    psi1[:,:,:,:,:,:] = x1
    psi2[:,:,:,:,:] = x2
    psi3[:,:,:,:,:] = x3
    output[:,:,:,:,:] = y
  
    f.close()

