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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks as cbks
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import layers

def loss_modified_biophysical(input_train_tm_tens,input_train_tp_tens,dfat_train_t_tens,te_train_t_tens,w_mean_r,w_std_r,f_mean_r,f_std_r,w_mean_i,w_std_i,f_mean_i,f_std_i,frq_mean,frq_std,r2_mean,r2_std,output_pred):

    pi = tf.constant(math.pi)
    num_echoes = 6

    dfat_train_t_c = tf.cast(tf.complex(dfat_train_t_tens,0*dfat_train_t_tens),tf.complex64)
    te_train_t_c = tf.cast(tf.complex(te_train_t_tens,0*te_train_t_tens),tf.complex64)
    pi_cmp = tf.cast(tf.complex(pi,0*pi), tf.complex64)

    relAmps = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048]
    num_peaks = 6
    fat_phasor = []
    fat_phasor_cos = []
    for peak in range(0,num_peaks):
        freq_peak = dfat_train_t_c[:,:,:,:,peak]
        relAmps_temp = relAmps[peak]
        relAmps_temp = tf.cast(tf.complex(relAmps_temp,0*relAmps_temp), tf.complex64) 
        temp_fat_phasor =  tf.math.multiply(relAmps_temp,tf.exp(-1j*2*pi_cmp*freq_peak*te_train_t_c))
        temp_fat_phasor_cos = tf.math.multiply(relAmps_temp,tf.math.cos(2*pi_cmp*freq_peak*te_train_t_c))
        if peak==0:
            fat_phasor = temp_fat_phasor
            fat_phasor_cos = temp_fat_phasor_cos
        else:
            fat_phasor = fat_phasor + temp_fat_phasor
            fat_phasor_cos = fat_phasor_cos + temp_fat_phasor_cos

    wat_r = output_pred[:,:,:,0]
    watt_r = k.expand_dims(wat_r,3)
    watt_r = k.repeat_elements(watt_r,num_echoes,3)
    watt_r = tf.scalar_mul(w_std_r,watt_r)+w_mean_r
    
    fat_r = output_pred[:,:,:,1]
    fatt_r = k.expand_dims(fat_r,3)
    fatt_r = k.repeat_elements(fatt_r,num_echoes,3)
    fatt_r = tf.scalar_mul(f_std_r,fatt_r)+f_mean_r

    r2 = output_pred[:,:,:,2]
    r2t = k.expand_dims(r2,3)
    r2t = k.repeat_elements(r2t,num_echoes,3)
    r2t = tf.scalar_mul(r2_std,r2t)+r2_mean

    wat_i = output_pred[:,:,:,3]
    watt_i = k.expand_dims(wat_i,3)
    watt_i = k.repeat_elements(watt_i,num_echoes,3)
    watt_i = tf.scalar_mul(w_std_i,watt_i)+w_mean_i

    fat_i = output_pred[:,:,:,4]
    fatt_i = k.expand_dims(fat_i,3)
    fatt_i = k.repeat_elements(fatt_i,num_echoes,3)
    fatt_i = tf.scalar_mul(f_std_i,fatt_i)+f_mean_i

    phase_err = output_pred[:,:,:,5]
    phase_err = tf.complex(phase_err,0*phase_err)
    phase_err = tf.cast(phase_err,tf.complex64)

    frq = output_pred[:,:,:,6]
    frqt = k.expand_dims(frq,3)
    frqt = k.repeat_elements(frqt,num_echoes,3)
    frqt = tf.scalar_mul(frq_std,frqt)+frq_mean

    watt_c =  tf.cast(tf.complex(watt_r,0*watt_r), tf.complex64)
    fatt_c =  tf.cast(tf.complex(fatt_r,0*fatt_r), tf.complex64)

    watt_ci =  tf.cast(tf.complex(watt_i,0*watt_i), tf.complex64)
    fatt_ci =  tf.cast(tf.complex(fatt_i,0*fatt_i), tf.complex64)

    r2t_c =  tf.cast(tf.complex(r2t,0*r2t), tf.complex64)
    frqt_c = tf.cast(tf.complex(frqt,0*frqt),tf.complex64)
    
    signal = tf.cast(((watt_c)*tf.exp(1j*watt_ci) + tf.exp(1j*fatt_ci)*(fatt_c)*fat_phasor)*tf.exp(-1*r2t_c*te_train_t_c)*tf.exp(-1j*2*pi_cmp*frqt_c*te_train_t_c),tf.complex64)
    signal = tf.cast(signal,tf.complex64)
    
    fat_phasor_mag = tf.cast(tf.math.square(tf.abs(fat_phasor)),tf.complex64)
    signal_mag = tf.math.square(watt_c) + tf.math.square(fatt_c)*fat_phasor_mag + 2*watt_c*fatt_c*fat_phasor_cos*tf.exp(-1*r2t_c*te_train_t_c)
    signal_mag = tf.math.sqrt(signal_mag)
    signal_mag = tf.cast(signal_mag,tf.complex64)

    input_train_t_mag = input_train_tm_tens[:,:,:,0:num_echoes]
    input_train_t_phs = input_train_tp_tens[:,:,:,0:num_echoes]

    gt_input_train2 = tf.cast(tf.multiply(tf.complex(input_train_t_mag,0*input_train_t_mag),tf.exp(1j*tf.complex(input_train_t_phs,0*input_train_t_phs))),tf.complex64)
  
    signal_diff = tf.abs(gt_input_train2-signal)

    echo1_sig_diff =  tf.abs((gt_input_train2[:,:,:,0]*tf.exp(1j*phase_err)) - signal[:,:,:,0])
  
    norm0 = tf.linalg.norm(echo1_sig_diff)
    norm1 = tf.linalg.norm(signal_diff[:,:,:,1])
    norm2 = tf.linalg.norm(signal_diff[:,:,:,2])
    norm3 = tf.linalg.norm(signal_diff[:,:,:,3])
    norm4 = tf.linalg.norm(signal_diff[:,:,:,4])
    norm5 = tf.linalg.norm(signal_diff[:,:,:,5])
    
    loss_phase = tf.sqrt(0.4*norm0 + 0.3*norm1 + 0.2*norm2 + 0.05*norm3 + 0.035*norm4 + 0.015*norm5)

    mag_only_loss = tf.sqrt(tf.linalg.norm( tf.cast(tf.math.abs(signal_mag),tf.float32) - input_train_t_mag))
    
    loss = (loss_phase) + 0.01*(mag_only_loss)
    
    return loss



