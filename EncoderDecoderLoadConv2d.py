

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
import glob
import os
import time


import h5py
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse

from util import log
from vqa_util import *

from base64 import decode
import tensorflow as tf
import numpy as np

 
import glob
import os
import time

import numpy as np

from input_ops import create_input_ops 
import argparse
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model

matplotlib.use('TkAgg') 

def encoded(inputs): 
  
  # Conv Block 1 -> BatchNorm->leaky Relu
  encoded = tf.keras.layers.Conv2D(30, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
  encoded = tf.keras.layers.BatchNormalization(name='batchnorm_1')(encoded)
  encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_1')(encoded)
  # Conv Block 2 -> BatchNorm->leaky Relu
  encoded = tf.keras.layers.Conv2D(15, kernel_size=3, strides= 4, padding='same', name='conv_2')(encoded)
  encoded = tf.keras.layers.BatchNormalization(name='batchnorm_2')(encoded)
  encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_2')(encoded)
  # Conv Block 3 -> BatchNorm->leaky Relu
  encoded = tf.keras.layers.Conv2D(8, kernel_size=3, strides=4, padding='same', name='conv_3')(encoded)
  encoded = tf.keras.layers.BatchNormalization(name='batchnorm_3')(encoded)
  encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_3')(encoded)

  encoded = tf.keras.layers.Conv2D(4, kernel_size=3, strides=2, padding='same', name='conv_4a')(encoded)
  encoded = tf.keras.layers.BatchNormalization(name='batchnorm_4a')(encoded)
  encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_4a')(encoded)
  
  
  return encoded



def decoded(inputs): 
    
    #Decoder
    # DeConv Block 1-> BatchNorm->leaky Relu
    decoded = tf.keras.layers.Conv2DTranspose(30, kernel_size=3, strides= 1, padding='same',name='conv_transpose_1')(inputs)
    decoded = tf.keras.layers.BatchNormalization(name='batchnorm_4')(decoded)
    decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_4')(decoded)
    # DeConv Block 2-> BatchNorm->leaky Relu
    decoded = tf.keras.layers.Conv2DTranspose(15,kernel_size= 3, strides= 4, padding='same', name='conv_transpose_2')(decoded)
    decoded = tf.keras.layers.BatchNormalization(name='batchnorm_5')(decoded)
    decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_5')(decoded)
    # DeConv Block 3-> BatchNorm->leaky Relu


    decoded = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides= 4, padding='same', name='conv_transpose_30')(decoded)
    decoded = tf.keras.layers.BatchNormalization(name='batchnorm_31')(decoded)
    decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_32')(decoded)

    decoded = tf.keras.layers.Conv2DTranspose(4, kernel_size=3, strides= 2, padding='same', name='conv_transpose_a')(decoded)
    decoded = tf.keras.layers.BatchNormalization(name='batchnorm_b')(decoded)
    decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_c')(decoded)

    # output
    outputs = tf.keras.layers.Conv2DTranspose(3, kernel_size= 3,strides= 1,padding='same', activation='sigmoid', name='conv_transpose_4a')(decoded)
    return outputs

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))
  
def loadWeights(modelLoad,neuralNetworkAdress,neuralweightAdress ):
  model1 = tf.keras.models.load_model(neuralNetworkAdress, compile=False) 
  model1.compile(loss=SSIMLoss, optimizer='adam', metrics=SSIMLoss)
  model1.summary()
  model1.load_weights(neuralweightAdress)
  for _layer in modelLoad.layers:
    try:
       modelLoad.get_layer(_layer.name).set_weights(model1.get_layer(_layer.name).get_weights())
       print(_layer.name)
    except:
       print("erro")
       print(_layer.name)
  return modelLoad

def loadDataBase() : 
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', type=int, default=50)
  parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
  parser.add_argument('--prefix', type=str, default='default')
  parser.add_argument('--checkpoint', type=str, default=None)
  parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_teste_decode-image')
  parser.add_argument('--learning_rate', type=float, default=2.5e-4)
  parser.add_argument('--lr_weight_decay', action='store_true', default=False)
  config = parser.parse_args()
  import sort_of_clevr as dataset
  path = os.path.join('./datasets', config.dataset_path)
  config.data_info = dataset.get_data_info()
  config.conv_info = dataset.get_conv_info()
  dataset_train, dataset_test = dataset.create_default_splits(path)
  batch_size = 128
  data_id = dataset_train.ids

  train_imgs = []

  test_imgs = []

  def load_fn(id,dataset):
    # image [n, n], q: [m], a: [l]
    img, q, a,imgDec,Reshap,Shape = dataset.get_data(id)
    return (id, img.astype(np.float32), q.astype(np.float32), a.astype(np.float32))


  for id in data_id:
    train_imgs.append(load_fn(id,dataset_train)[1])

  data_id = dataset_test.ids

  for id in data_id:
    test_imgs.append(load_fn(id,dataset_test)[1])

  x_test = np.array(test_imgs)
  x_train =  np.array(train_imgs)

  #(x_train, _), (x_test, _) =  tf.keras.datasets.fashion_mnist.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))
  return x_test,x_train

def printResultTest(decoded_imgs,x_test ):
  n = 5
  plt.figure(figsize=(20, 4))
  for i in range(1, n + 1):
      # Display original
      ax = plt.subplot(2, n, i)
      plt.imshow(x_test[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      plt.title("Original")
      ax.get_yaxis().set_visible(False)
      # Display reconstruction
      ax = plt.subplot(2, n, i + n)
      plt.title("Reconstructed")
      plt.imshow(decoded_imgs[i])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
  plt.show()  
  #plt.savefig("C:\Source\Relation-Network-Tensorflow\decoder\teste.png") 


inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer')
autoencoded = Model(inputs,encoded(inputs))
autoencoded.compile(loss=SSIMLoss, optimizer='adam', metrics=SSIMLoss)
autoencoded.summary()
loadWeights(autoencoded,"C:\Source\PesquisaMestrado\decoder1\model3","C:\Source\PesquisaMestrado\decoder1\weights4" )

x_test,train_imgs = loadDataBase() 


#sleep(100) # sleep to keras unload the objetcs
imputDecoder = tf.keras.Input(shape=(4, 4, 4), name='input_layer_dec')
autodecoder = Model(imputDecoder,decoded(imputDecoder))
autodecoder.compile(loss=SSIMLoss, optimizer='adam', metrics=SSIMLoss)
autodecoder.summary()
loadWeights(autodecoder,"C:\Source\PesquisaMestrado\decoder1\model3","C:\Source\PesquisaMestrado\decoder1\weights4" )



encode = autoencoded.predict(x_test)
decode = autodecoder.predict(encode)


printResultTest(decode,x_test )

a=1




