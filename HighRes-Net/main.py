import os
import math
import string
import sys
import tensorflow as tf
import scipy.io as sio
import numpy as np
import time
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, Lambda, UpSampling2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as k
import tensorflow.keras.callbacks as cbks
from tensorflow.keras.callbacks import CSVLogger
from model import *
from loss_modified_biophysical import *

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

num_echos = 6

Input_path = sys.argv[1]
path_mat_out = sys.argv[2]

Input_path = Input_path+"High-Resolution mGRE"

#========load data==========

path = os.path.join(Input_path,"final_images.npy")
final_images = np.load(path)

path = os.path.join(Input_path,"final_masks.npy")
final_masks = np.load(path)

path = os.path.join(Input_path,"final_TE_masks.npy")
final_TE_masks = np.load(path)

path = os.path.join(Input_path,"final_Freq_masks.npy")
final_Freq_masks = np.load(path)

batch_no = "HighRes-Net"

input_train_t = final_images[:,:,:,:] 
input_test_t = final_images[:,:,:,:]

te_train_t = final_TE_masks[:,:,:,:]  
te_test_t = final_TE_masks[:,:,:,:]  

dfat_train_t = final_Freq_masks[:,:,:,:]
dfat_test_t = final_Freq_masks[:,:,:,:]

mask_test_t = final_masks[:,:,:]

#Normalizing the magnitude component between 0 to 1
for itr in range(0,final_images.shape[0]):
    img = final_images[itr,:,:,:]
    
    for i in range(0,num_echos):
        temp = img[:,:,i]
        temp = (temp - temp.min()) / (temp.max()- temp.min())
        final_images[itr,:,:,i] = temp
        

print("Input Data Dimensions:")
print(input_train_t.shape)     
print(te_train_t.shape)      
print(dfat_train_t.shape)    
print("\n")

#======z-score calculated from average of multiple datasets=========
w_mean_r = float(0.2)
w_std_r = float(0.1)
f_mean_r = float(0.1) 
f_std_r = float(0.2)
frq_mean = float(30) 
frq_std = float(30)
r2_mean = float(80)
r2_std = float(50)
w_mean_i = float(-1)
w_std_i = float(1) 
f_mean_i =float(-1)
f_std_i = float(1)

input_train_tm = input_train_t[:,:,:,0:num_echos]
input_train_tp = input_train_t[:,:,:,num_echos:num_echos*2]
input_test_tm = input_test_t[:,:,:,0:num_echos]
input_test_tp = input_test_t[:,:,:,num_echos:num_echos*2]

#==========================network============================
input_shape = (np.squeeze(input_train_tm[:1, :,:, :])).shape
input_train_tm_tens = Input(input_shape, name='input_train_tm')
input_train_tp_tens = Input(input_shape, name='input_train_tp')
input_shape = (np.squeeze(dfat_train_t[:1, :,:, :])).shape
dfat_train_t_tens = Input(input_shape, name='dfat_train_t')
input_shape = (np.squeeze(te_train_t[:1, :,:, :])).shape
te_train_t_tens = Input(input_shape, name='te_train_t')
output_train_t_tens = Input(input_shape, name='output_train_t')
model,output_pred = network(input_train_tm_tens,input_train_tp_tens, dfat_train_t_tens, te_train_t_tens)

#==========define loss functions: loss1 for UTD/NTD, loss2 for STD==========
filepath = "batch_"+str(batch_no)+".hdf5"
callbacks = [ReduceLROnPlateau(monitor="loss",factor=0.1, patience=10, min_lr=(0.0002), verbose=1), ModelCheckpoint(filepath,monitor="loss", verbose=1, save_best_only=True, save_weights_only=True, mode='auto')]

#==========compile/train/test==========
custom_loss4 = model.add_loss(1*loss_modified_biophysical(input_train_tm_tens,input_train_tp_tens,dfat_train_t_tens,te_train_t_tens,w_mean_r,w_std_r,f_mean_r,f_std_r,w_mean_i,w_std_i,f_mean_i,f_std_i,frq_mean,frq_std,r2_mean,r2_std,output_pred))

model.compile(optimizer=Adam(learning_rate=(0.006)), loss=custom_loss4)


history = model.fit([input_train_tm,input_train_tp, dfat_train_t, te_train_t], callbacks=callbacks, batch_size=10, epochs=15000, shuffle=True,verbose=1)

print("Loading the model:")
filepath = "batch_"+str(batch_no)+".hdf5"
model.load_weights(filepath)

print("Prediction using the model:")
outresults = model.predict([input_test_tm,input_test_tp, dfat_test_t, te_test_t], batch_size=2)

#==========save results==========
outresults[:,:,:,0] = outresults[:,:,:,0]*w_std_r+w_mean_r
outresults[:,:,:,1] = outresults[:,:,:,1]*f_std_r+f_mean_r
outresults[:,:,:,6] = outresults[:,:,:,6]*frq_std+frq_mean 
outresults[:,:,:,2] = outresults[:,:,:,2]*r2_std+r2_mean
sio.savemat(path_mat_out,{'test_pd':outresults,'w_mean_r':w_mean_r,'w_std_r':w_std_r,'f_mean_r':f_mean_r,'f_std_r':f_std_r,'frq_mean':frq_mean,'frq_std':frq_std,'r2_mean':r2_mean,'r2_std':r2_std,'w_mean_i':w_mean_i,'w_std_i':w_std_i,'f_mean_i':f_mean_i,'f_std_i':f_std_i,'mask':mask_test_t})
