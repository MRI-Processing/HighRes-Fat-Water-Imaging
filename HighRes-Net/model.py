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
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, Flatten, Lambda, UpSampling2D
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.layers import concatenate, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import keras.backend as k
import keras.callbacks as cbks
from tensorflow.keras.regularizers import l1
from keras.callbacks import CSVLogger

def network(input_train_tm_tens,input_train_tp_tens, dfat_train_t_tens, te_train_t_tens):

 n_filters = 32
 kernel_size =2
 dropout_rate=0.1
 reg_wt = 0.000001
 reg_wt_F = 0.000001
 dropout_rate_F = 0.1

 c1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_uniform', kernel_regularizer=l1(reg_wt))(input_train_tm_tens)
 c1 = BatchNormalization()(c1)
 #c1 = Dropout(dropout_rate)(c1)
 p1 = MaxPooling2D((2, 2)) (c1)

 c2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l1(reg_wt)) (p1)
 c2 = BatchNormalization()(c2)
 #c2 = Dropout(dropout_rate)(c2)
 p2 = MaxPooling2D((2, 2))(c2)

 c3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (p2)
 c3 = BatchNormalization()(c3)
 #c3 = Dropout(dropout_rate)(c3)
 p3 = MaxPooling2D((2, 2)) (c3)

 c4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (p3)
 c4 = BatchNormalization()(c4)
 #c4 = Dropout(dropout_rate)(c4)
 p4 = MaxPooling2D((2, 2)) (c4)

 c5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform',padding='same',kernel_regularizer=l1(reg_wt)) (p4)
 c5 = BatchNormalization()(c5)
 #c5 = Dropout(dropout_rate)(c5)
 p5 = MaxPooling2D((2, 2)) (c5)

 c6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (p5)
 c6 = BatchNormalization()(c6)
 #c6 = Dropout(dropout_rate)(c6)
 c6 = UpSampling2D(size=(2, 2), data_format=None) (c6)

 u7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (c6)
 u7 = Add()([u7, c5])
 c7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (u7)
 c7 = BatchNormalization()(c7)
 #c7 = Dropout(dropout_rate)(c7)
 c7 = UpSampling2D(size=(2, 2), data_format=None) (c7)

 u8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (c7)
 u8 = Add()([u8, c4])
 c8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (u8)
 c8 = BatchNormalization()(c8)
 #c8 = Dropout(dropout_rate)(c8)
 c8 = UpSampling2D(size=(2, 2), data_format=None) (c8)


 u9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (c8)
 u9 = Add()([u9, c3])
 c9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (u9)
 c9 = BatchNormalization()(c9)
 #c9 = Dropout(dropout_rate)(c9)
 c9 = UpSampling2D(size=(2, 2), data_format=None) (c9)
 
 u10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (c9)
 u10 = Add()([u10, c2])
 c10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (u10)
 c10 = BatchNormalization()(c10)
 #c10 = Dropout(dropout_rate)(c10)
 c10 = UpSampling2D(size=(2, 2), data_format=None) (c10)

 u11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (c10)
 u11 = Add()([u11, c1])
 c11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (u11)
 c11 = BatchNormalization()(c11)
 #c11 = Dropout(dropout_rate)(c11)
 c11 = UpSampling2D(size=(1, 1), data_format=None) (c11)

 output_pred1 = Conv2D(3,(1, 1), activation='linear') (c11)


 cc1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_uniform', kernel_regularizer=l1(reg_wt))(input_train_tp_tens)
 cc1 = BatchNormalization()(cc1)
 #cc1 = Dropout(dropout_rate)(cc1)
 pp1 = MaxPooling2D((2, 2)) (cc1)

 cc2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l1(reg_wt)) (pp1)
 cc2 = BatchNormalization()(cc2)
 #cc2 = Dropout(dropout_rate)(cc2)
 pp2 = MaxPooling2D((2, 2))(cc2)

 cc3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (pp2)
 cc3 = BatchNormalization()(cc3)
 #cc3 = Dropout(dropout_rate)(cc3)
 pp3 = MaxPooling2D((2, 2)) (cc3)

 cc4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (pp3)
 cc4 = BatchNormalization()(cc4)
 #cc4 = Dropout(dropout_rate)(cc4)
 pp4 = MaxPooling2D((2, 2)) (cc4)

 cc5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform',padding='same',kernel_regularizer=l1(reg_wt)) (pp4)
 cc5 = BatchNormalization()(cc5)
 #cc5 = Dropout(dropout_rate)(cc5)
 pp5 = MaxPooling2D((2, 2)) (cc5)

 cc6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (pp5)
 cc6 = BatchNormalization()(cc6)
 #cc6 = Dropout(dropout_rate)(cc6)
 cc6 = UpSampling2D(size=(2, 2), data_format=None) (cc6)

 uu7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (cc6)
 uu7 = Add()([uu7, cc5])
 cc7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (uu7)
 cc7 = BatchNormalization()(cc7)
 #cc7 = Dropout(dropout_rate)(cc7)
 cc7 = UpSampling2D(size=(2, 2), data_format=None) (cc7)

 uu8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (cc7)
 uu8 = Add()([uu8, cc4])
 cc8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (uu8)
 cc8 = BatchNormalization()(cc8)
 #cc8 = Dropout(dropout_rate)(cc8)
 cc8 = UpSampling2D(size=(2, 2), data_format=None) (cc8)


 uu9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (cc8)
 uu9 = Add()([uu9, cc3])
 cc9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (uu9)
 cc9 = BatchNormalization()(cc9)
 #cc9 = Dropout(dropout_rate)(cc9)
 cc9 = UpSampling2D(size=(2, 2), data_format=None) (cc9)
 
 uu10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (cc9)
 uu10 = Add()([uu10, cc2])
 cc10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (uu10)
 cc10 = BatchNormalization()(cc10)
 #cc10 = Dropout(dropout_rate)(cc10)
 cc10 = UpSampling2D(size=(2, 2), data_format=None) (cc10)

 uu11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt)) (cc10)
 uu11 = Add()([uu11, cc1])
 cc11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt)) (uu11)
 cc11 = BatchNormalization()(cc11)
 #cc11 = Dropout(dropout_rate)(cc11)
 cc11 = UpSampling2D(size=(1, 1), data_format=None) (cc11)
 
 output_pred2 = Conv2D(3,(1, 1), activation='linear') (cc11)

 ###_____ Field Inhomogenity Network starts_____
 ccc1 = Conv2D(12, kernel_size=(kernel_size, kernel_size), activation = 'sigmoid', padding = 'same', kernel_initializer = 'glorot_uniform', kernel_regularizer=l1(reg_wt_F))(input_train_tp_tens)
 ccc1 = BatchNormalization()(ccc1)
 ccc1 = Dropout(dropout_rate_F)(ccc1)
 ppp1 = MaxPooling2D((2, 2)) (ccc1)

 ccc2 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same', kernel_regularizer=l1(reg_wt_F)) (ppp1)
 ccc2 = BatchNormalization()(ccc2)
 ccc2 = Dropout(dropout_rate_F)(ccc2)
 ppp2 = MaxPooling2D((2, 2))(ccc2)

 ccc3 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (ppp2)
 ccc3 = BatchNormalization()(ccc3)
 ccc3 = Dropout(dropout_rate_F)(ccc3)
 ppp3 = MaxPooling2D((2, 2)) (ccc3)

 ccc4 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (ppp3)
 ccc4 = BatchNormalization()(ccc4)
 ccc4 = Dropout(dropout_rate_F)(ccc4)
 ppp4 = MaxPooling2D((2, 2)) (ccc4)

 ccc5 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform',padding='same',kernel_regularizer=l1(reg_wt_F)) (ppp4)
 ccc5 = BatchNormalization()(ccc5)
 ccc5 = Dropout(dropout_rate_F)(ccc5)
 ppp5 = MaxPooling2D((2, 2)) (ccc5)

 ccc6 = Conv2D(384,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (ppp5)
 ccc6 = BatchNormalization()(ccc6)
 ccc6 = Dropout(dropout_rate_F)(ccc6)
 ccc6 = UpSampling2D(size=(2, 2), data_format=None) (ccc6)

 uuu7 = Conv2DTranspose(192,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt_F)) (ccc6)
 uuu7 = Add()([uuu7, ccc5])
 ccc7 = Conv2D(192,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (uuu7)
 ccc7 = BatchNormalization()(ccc7)
 ccc7 = Dropout(dropout_rate_F)(ccc7)
 ccc7 = UpSampling2D(size=(2, 2), data_format=None) (ccc7)

 uuu8 = Conv2DTranspose(96,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt_F)) (ccc7)
 uuu8 = Add()([uuu8, ccc4])
 ccc8 = Conv2D(96,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (uuu8)
 ccc8 = BatchNormalization()(ccc8)
 ccc8 = Dropout(dropout_rate_F)(ccc8)
 ccc8 = UpSampling2D(size=(2, 2), data_format=None) (ccc8)

 uuu9 = Conv2DTranspose(48,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt_F)) (ccc8)
 uuu9 = Add()([uuu9, ccc3])
 ccc9 = Conv2D(48,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (uuu9)
 ccc9 = BatchNormalization()(ccc9)
 ccc9 = Dropout(dropout_rate_F)(ccc9)
 ccc9 = UpSampling2D(size=(2, 2), data_format=None) (ccc9)
 
 uuu10 = Conv2DTranspose(24,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt_F)) (ccc9)
 uuu10 = Add()([uuu10, ccc2])
 ccc10 = Conv2D(24,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (uuu10)
 ccc10 = BatchNormalization()(ccc10)
 ccc10 = Dropout(dropout_rate_F)(ccc10)
 ccc10 = UpSampling2D(size=(2, 2), data_format=None) (ccc10)

 uuu11 = Conv2DTranspose(12,kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same',kernel_regularizer=l1(reg_wt_F)) (ccc10)
 uuu11 = Add()([uuu11, ccc1])
 ccc11 = Conv2D(12,kernel_size=(kernel_size, kernel_size), activation='sigmoid', kernel_initializer='glorot_uniform', padding='same',kernel_regularizer=l1(reg_wt_F)) (uuu11)
 ccc11 = BatchNormalization()(ccc11)
 ccc11 = Dropout(dropout_rate_F)(ccc11)
 ccc11 = UpSampling2D(size=(1, 1), data_format=None) (ccc11)
 output_pred3 = Conv2D(1,(1, 1), activation='linear') (ccc11)

 output_pred = concatenate([output_pred1, output_pred2, output_pred3])

 model = Model(inputs=[input_train_tm_tens, input_train_tp_tens, dfat_train_t_tens, te_train_t_tens], outputs=[output_pred])

 return model,output_pred


