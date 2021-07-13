import tensorflow as tf
import numpy as np
import keras.backend as kb
from keras.optimizers import Adam
import os 
from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout,Conv2DTranspose,Add
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed,Activation
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file


def OcclusionNet(input_height=7,input_width=7,input_channel=512, conv_output_channel=512):
  input=Input(shape=(input_height,input_width,input_channel))

#   convolution
  x = Conv2D(conv_output_channel, (3, 3), activation='relu', padding='same', name='block1_conv1')(input)
  x = Conv2D(conv_output_channel, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
  x = Conv2D(conv_output_channel, (3, 3), activation='relu', padding='same', name='block1_conv3')(x)

  o=Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='block1_conv4')(x)
  


  output=o
  model=Model(input,output)
  return model