
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import matplotlib.pyplot as plt
from util import log
from pprint import pprint

from model_rn_image_conv2d import Autoencoder
from model_rn_image_conv2d import Autoencoder2
from input_ops import create_input_ops 

import os
import time
import tensorflow as tf
import tf_slim as slim
import numpy as np


class Trainer(object):

   

    def __init__(self,
                 config,
                 dataset,
                 dataset_test):
        self.config = config
        hyper_parameter_str = config.dataset_path+'_lr_'+str(config.learning_rate)
        self.train_dir = './train_dir/%s-%s-%s-%s' % (
            config.model,
            config.prefix,
            hyper_parameter_str,
            time.strftime("%Y%m%d-%H%M%S")
        )
        #tf.compat.v1.enable_eager_execution()
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        log.infov("Train Dir: %s", self.train_dir)

        # --- input ops ---
        self.batch_size = config.batch_size

        _, self.batch_train, self.img_train = create_input_ops(dataset, self.batch_size,
                                               is_training=True)
        _, self.batch_test, self.img_test = create_input_ops(dataset_test, self.batch_size,
                                              is_training=False)
       


        Autoencoder
     

    

    def train(self):

        log.infov("Training Starts!")
        print(self.batch_train)
        #alterei aqui 

        x_train = self.img_train.astype('float32') / 255.
        x_test = self.img_test.astype('float32') / 255.



        x_train = self.img_train
        x_test = self.img_test 
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        # Tensorflow implementation
        autoencodertf = Autoencoder2(x_train.shape[1], 32)
        autoencodertf.train(x_train, x_test, 1, 1)
        encoded_img = autoencodertf.getEncodedImage(x_test[1])
        decoded_img = autoencodertf.getDecodedImage(x_test[1])        
        plt.figure(figsize=(20, 4))
        subplot = plt.subplot(2, 10, 1)

        plt.imshow(x_test[1].reshape(28, 28))
        plt.gray()
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)

        subplot = plt.subplot(2, 10, 2)
        plt.imshow(decoded_img.reshape(28, 28))
        plt.gray()
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        #autoencodertf = Autoencoder2(self.img_train[0].shape, 32)
        #autoencodertf.train(self.img_train, self.img_test, 100, 100)

        autoencoder = Autoencoder()
        

        autoencoder.train(  self.img_train,self.img_test, 2, 2)

def check_data_path(path):
    if os.path.isfile(os.path.join(path, 'data.hy')) \
           and os.path.isfile(os.path.join(path, 'id.txt')):
        return True
    else:
        return False

def main():
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

    path = os.path.join('./datasets', config.dataset_path)

    if check_data_path(path):
        import sort_of_clevr as dataset
    else:
        raise ValueError(path)

    config.data_info = dataset.get_data_info()
    config.conv_info = dataset.get_conv_info()
    dataset_train, dataset_test = dataset.create_default_splits(path)

    trainer = Trainer(config,
                      dataset_train, dataset_test)

    log.warning("dataset: %s, learning_rate: %f",
                config.dataset_path, config.learning_rate)
    trainer.train()
    

if __name__ == '__main__':

    main()
