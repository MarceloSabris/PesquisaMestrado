
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import matplotlib.pyplot as plt
from util import log
from pprint import pprint

from input_ops import create_input_ops

import os
import time
import tensorflow as tf
import tf_slim as slim
import numpy
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

class Trainer(object):

    def Training()
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))
      model.add(layers.MaxPooling2D((2, 2)))
      model.add(layers.Conv2D(64, (3, 3), activation='relu'))

      model.add(layers.Flatten())
      model.add(layers.Dense(64, activation='relu'))
      model.add(layers.Dense(10))


      model.summary()

      model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

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

        log.warning("dataset: %s, learning_rate: %f",
                    config.dataset_path, config.learning_rate)
        
if __name__ == '__main__':
    main()
