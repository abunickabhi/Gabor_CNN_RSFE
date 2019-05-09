"""
Created on Wed May  8 13:00:14 2019

@author: abhi
"""

from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Merge
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K

def _2DCNN(img,gt):
    my_input = Input(shape=input_shape, dtype='float32')
            
            conv_1 = Convolution2D(64,
                                   3,
                                   3,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_1 = MaxPooling2D(pool_size=(2,2),
                                         dim_ordering=dim_ordering
                                         )(conv_1)

            pooled_conv_1_dropped = Dropout(drop_rate)(pooled_conv_1)
            
            conv_11 = Convolution2D(96,
                                    3,
                                    3,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_1_dropped)
            
            pooled_conv_11 = MaxPooling2D(pool_size=(2,2),
                                          dim_ordering=dim_ordering
                                          )(conv_11)
                                          
            pooled_conv_11_dropped = Dropout(drop_rate)(pooled_conv_11)
            pooled_conv_11_dropped_flat = Flatten()(pooled_conv_11_dropped)

            conv_2 = Convolution2D(64,
                                   4,
                                   4, 
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_2 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_2)
            pooled_conv_2_dropped = Dropout(drop_rate)(pooled_conv_2)
            
            conv_22 = Convolution2D(96,
                                    4,
                                    4, 
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering,
                                    )(pooled_conv_2_dropped)
            
            pooled_conv_22 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_22)
            pooled_conv_22_dropped = Dropout(drop_rate)(pooled_conv_22)
            pooled_conv_22_dropped_flat = Flatten()(pooled_conv_22_dropped)

            conv_3 = Convolution2D(64,
                                   5,
                                   5,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_3 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_3)
            pooled_conv_3_dropped = Dropout(drop_rate)(pooled_conv_3)
            
            conv_33 = Convolution2D(96,
                                    5,
                                    5,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_3_dropped)
            
            pooled_conv_33 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_33)
            pooled_conv_33_dropped = Dropout(drop_rate)(pooled_conv_33)
            pooled_conv_33_dropped_flat = Flatten()(pooled_conv_33_dropped)                        
            
            conv_4 = Convolution2D(64,
                                   6,
                                   6,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_4 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_4)
            pooled_conv_4_dropped = Dropout(drop_rate)(pooled_conv_4)
            
            conv_44 = Convolution2D(96,
                                    6,
                                    6,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_4_dropped)
            
            pooled_conv_44 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering) (conv_44)
            pooled_conv_44_dropped = Dropout(drop_rate) (pooled_conv_44)
            pooled_conv_44_dropped_flat = Flatten()(pooled_conv_44_dropped)

            merge = Merge(mode='concat')([pooled_conv_11_dropped_flat,
                                          pooled_conv_22_dropped_flat,
                                          pooled_conv_33_dropped_flat,
                                          pooled_conv_44_dropped_flat])
            
            merge_dropped = Dropout(drop_rate)(merge)
            
            dense = Dense(128,
                          activation='relu'
                          )(merge_dropped)
            
            dense_dropped = Dropout(drop_rate)(dense)
            
            prob = Dense(output_dim=num_classes,
                         activation='softmax'
                         )(dense_dropped)
            