import glob
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

import tfplot
from tensorflow import keras

from input_ops import create_input_ops 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--model', type=str, default='rn', choices=['rn', 'baseline'])
parser.add_argument('--prefix', type=str, default='default')
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--dataset_path', type=str, default='Sort-of-CLEVR_default')
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
  img, q, a = dataset.get_data(id)
  return (id, img.astype(np.float32), q.astype(np.float32), a.astype(np.float32))


for id in data_id:
  train_imgs.append(load_fn(id,dataset_train)[1])

data_id = dataset_test.ids

for id in data_id:
  test_imgs.append(load_fn(id,dataset_test)[1])



       
a="" 
batch_train = "" 
b=""
batch_test=""
x_test = np.array(test_imgs)
x_train =  np.array(train_imgs)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
#x_test = x_test.astype('float32')
#x_test = x_test / 255.
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).\
shuffle(60000).batch(128)
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)

normalized_ds = train_dataset.map(lambda x: normalization_layer(x))
image_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image)) 

input_encoder = (128, 128, 3)
input_decoder = (2,)

def encoder(input_encoder):
    
    inputs = keras.Input(shape=input_encoder, name='input_layer')
    x = layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    #x = layers.Dropout(rate = 0.25)(x)

    x = layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    flatten = layers.Flatten()(x)
    bottleneck = layers.Dense(2, name='dense_1')(flatten)
    model = tf.keras.Model(inputs, bottleneck, name="Encoder")
    return model
enc = encoder(input_encoder)

enc.summary()

def decoder(input_decoder):
    
    inputs = keras.Input(shape=input_decoder, name='input_layer')
    x = layers.Dense(65536, name='dense_1')(inputs)
    #x = tf.reshape(x, [-1, 7, 7, 64], name='Reshape_Layer')
    x = layers.Reshape((32,32,64), name='Reshape_Layer')(x)
    x = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    outputs = layers.Conv2DTranspose(1, 3, 1,padding='same', activation='sigmoid', name='conv_transpose_4')(x)
    model = tf.keras.Model(inputs, outputs, name="Decoder")
    return model

dec = decoder(input_decoder)

dec.summary()
def generate_and_save_images(model,  test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    latent = enc(test_input, training=False)
    predictions = dec(latent, training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(1,1))

    #x = plt.subplot()
    plt.imshow(predictions[0])
    plt.axis("off")
    normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
    plt.show()
    fig = plt.figure(figsize=(1,1))

    #x = plt.subplot()
    plt.imshow(test_input[0])
    plt.axis("off")
    normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
    plt.show()


index = np.random.choice(range(len(x_test)), 2)
example_images = x_test[index]
generate_and_save_images([enc,dec],example_images)
latent = enc(example_images, training=False)
predictions = dec(latent, training=False)
print(predictions.shape)
fig = plt.figure(figsize=(1,1))

    #x = plt.subplot()
plt.imshow(example_images[0])
plt.axis("off")
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
plt.show()


fig = plt.figure(figsize=(1,1))

    #x = plt.subplot()
plt.imshow(predictions[1])
plt.axis("off")
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
plt.show()