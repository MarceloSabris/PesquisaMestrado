from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import numpy as np
from PIL import Image, ImageDraw
import os
import argparse

from util import log
from vqa_util import *
import progressbar
from base64 import decode
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
import glob
import os
import time

import numpy as np

from input_ops import create_input_ops 
import argparse
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model


class Representation:

    def __init__(self, x, y, color, shape):
        self.x = x
        self.y = y
        self.color = color
        self.shape = shape

    def print_graph(self):
        for i in range(len(self.x)):
            s = 'circle' if self.shape[i] else 'rectangle'
            print('{} {} at ({}, {})'.format(color2str(self.color[i]),
                                             s, self.x[i], self.y[i]))

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





def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))

def GenerateEnconder(img): 
    inputs = tf.keras.Input(img.shape(), name='input_layer')
    autoencoder = tf.keras.Model(inputs, encoded(inputs))
    optimizer = tf.keras.optimizers.Adam(lr = 0.017)
    autoencoder.compile(optimizer=optimizer, loss=SSIMLoss)
    autoencoder.summary()
    return autoencoder

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
    
def creatModelDecode(imputShape):
    inputs = tf.keras.Input(imputShape, name='input_layer')
    autoencoded = Model(inputs,encoded(inputs))
    autoencoded.compile(loss=SSIMLoss, optimizer='adam', metrics=SSIMLoss)
    autoencoded.summary()
    return loadWeights(autoencoded,"C:\Source\PesquisaMestrado\decoder1\model3","C:\source\PesquisaMestrado\decoder1\weights4" )
     




def generator(config):
    img_size = config.img_size
    dataset_size = config.dataset_size
    dir_name = config.dir_name

    block_size = int(img_size*0.9/N_GRID)
    shape_size = int((img_size*0.9/N_GRID)*0.7/2)

    def generate_sample(img_size):
        # Generate I: [img_size, img_size, 3]
        img = Image.new('RGB', (img_size, img_size), color=BG_COLOR)
        drawer = ImageDraw.Draw(img)
        idx_coor = np.arange(N_GRID*N_GRID)
        np.random.shuffle(idx_coor)
        idx_color_shape = np.arange(NUM_COLOR)
        np.random.shuffle(idx_color_shape)
        coin = np.random.rand(NUM_SHAPE)
        X = []
        Y = []
        for i in range(NUM_SHAPE):
            x = idx_coor[i] % N_GRID
            y = (N_GRID - np.floor(idx_coor[i] / N_GRID) - 1).astype(np.uint8)
            # sqaure terms are added to remove ambiguity of distance
            position = ((x+0.5)*block_size-shape_size+x**2, (y+0.5)*block_size-shape_size+y**2,
                        (x+0.5)*block_size+shape_size+x**2, (y+0.5)*block_size+shape_size+y**2)
            X.append((x+0.5)*block_size+x**2)
            Y.append((y+0.5)*block_size+y**2)
            if coin[i] < 0.5:
                drawer.ellipse(position, fill=COLOR[idx_color_shape[i]])
            else:
                drawer.rectangle(position, fill=COLOR[idx_color_shape[i]])

        # Generate its representation
        color = idx_color_shape[:NUM_SHAPE]
        shape = coin < 0.5
        rep = Representation(np.stack(X).astype(np.int),
                             np.stack(Y).astype(np.int), color, shape)
        return np.array(img), rep

    def generate_question(rep):
        # Generate questions: [# of shape * # of Q, # of color + # of Q]
        Q = np.zeros((NUM_SHAPE*NUM_Q, NUM_COLOR+NUM_Q), dtype=np.bool)
        for i in range(NUM_SHAPE):
            v = np.zeros(NUM_COLOR)
            v[rep.color[i]] = True
            Q[i*NUM_Q:(i+1)*NUM_Q, :NUM_COLOR] = np.tile(v, (NUM_Q, 1))
            Q[i*NUM_Q:(i+1)*NUM_Q, NUM_COLOR:] = np.diag(np.ones(NUM_Q))
        return Q

    def generate_answer(rep):
        # Generate answers: [# of shape * # of Q, # of color + 4]
        # # of color + 4: [color 1, color 2, ... , circle, rectangle, yes, no]
        A = np.zeros((NUM_SHAPE*NUM_Q, NUM_COLOR+4), dtype=np.bool)
        for i in range(NUM_SHAPE):
            # Q1: circle or rectangle?
            if rep.shape[i]:
                A[i*NUM_Q, NUM_COLOR] = True
            else:
                A[i*NUM_Q, NUM_COLOR+1] = True

            # Q2: bottom?
            if rep.y[i] > int(img_size/2):
                A[i*NUM_Q+1, NUM_COLOR+2] = True
            else:
                A[i*NUM_Q+1, NUM_COLOR+3] = True

            # Q3: left?
            if rep.x[i] < int(img_size/2):
                A[i*NUM_Q+2, NUM_COLOR+2] = True
            else:
                A[i*NUM_Q+2, NUM_COLOR+3] = True

            distance = 1.1*(rep.y - rep.y[i]) ** 2 + (rep.x - rep.x[i]) ** 2
            idx = distance.argsort()
            # Q4: the color of the nearest object
            min_idx = idx[1]
            A[i*NUM_Q+3, rep.color[min_idx]] = True
            # Q5: the color of the farthest object
            max_idx = idx[-1]
            A[i*NUM_Q+4, rep.color[max_idx]] = True
        return A

    # output files
    f = h5py.File(os.path.join(dir_name, 'data.hy'), 'w')
    id_file = open(os.path.join(dir_name, 'id.txt'), 'w')

    # progress bar
    bar = progressbar.ProgressBar(maxval=100,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                           progressbar.Percentage()])
    bar.start()

    count = 0
    encoded = creatModelDecode((128,128,3))
    encoded = loadWeights(encoded,"C:\Source\PesquisaMestrado\decoder\model8","C:\Source\PesquisaMestrado\decoder\weights8" )
     
    while(1):
        I, R = generate_sample(config.img_size)
        A = generate_answer(R)
        Q = generate_question(R)
        E = encoded.predict(np.reshape(I/255., (1,128, 128, 3)))
        E = E[0]
        for j in range(NUM_SHAPE*NUM_Q):
            id = '{}'.format(count)
            id_file.write(id+'\n')
            grp = f.create_group(id)
            grp['image'] = I
            grp['question'] = Q[j, :]
            grp['answer'] = A[j, :]
            grp['encoded'] = E
            count += 1
            if count % (dataset_size / 100) == 0:
                bar.update(count / (dataset_size / 100))
            if count >= dataset_size:
                bar.finish()
                f.close()
                id_file.close()
                log.info('Dataset generated under {} with {} samples.'
                         .format(dir_name, dataset_size))
                return


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir_name', type=str,default='Sort-of-CLEVR_default')
    parser.add_argument('--dataset_size', type=int, default=50000)
    
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()

    basepath = './datasets'
    check_path(basepath)
    path = os.path.join(basepath, args.dir_name)
    check_path(path)
    args.dir_name = path

    generator(args)


if __name__ == '__main__':
    main()
