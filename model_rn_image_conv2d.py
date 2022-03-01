from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
import tf_slim  as slim
try:
    import tfplot
except:
    pass
from PIL import Image, ImageDraw
from ops import conv2d, fc
from util import log
import numpy as np
from vqa_util import question2str, answer2str
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import os

class Autoencoder2(object):
    def __init__(self, inout_dim, encoded_dim):
        learning_rate = 0.1 
        
        # Weights and biases
        hiddel_layer_weights = tf.Variable(tf.random_normal([inout_dim, encoded_dim]))
        hiddel_layer_biases = tf.Variable(tf.random_normal([encoded_dim]))
        output_layer_weights = tf.Variable(tf.random_normal([encoded_dim, inout_dim]))
        output_layer_biases = tf.Variable(tf.random_normal([inout_dim]))
        
        # Neural network
        self._input_layer = tf.placeholder('float', [None, inout_dim])
        self._hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(self._input_layer, hiddel_layer_weights), hiddel_layer_biases))
        self._output_layer = tf.matmul(self._hidden_layer, output_layer_weights) + output_layer_biases
        self._real_output = tf.placeholder('float', [None, inout_dim])
        
        self._meansq = tf.reduce_mean(tf.square(self._output_layer - self._real_output))
        self._optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self._meansq)
        self._training = tf.global_variables_initializer()
        self._session = tf.Session()
        
    def train(self, input_train, input_test, batch_size, epochs):
        self._session.run(self._training)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(input_train.shape[0]/batch_size)):
                epoch_input = input_train[ i * batch_size : (i + 1) * batch_size ]
                _, c = self._session.run([self._optimizer, self._meansq], feed_dict={self._input_layer: epoch_input, self._real_output: epoch_input})
                epoch_loss += c
                print('Epoch', epoch, '/', epochs, 'loss:',epoch_loss)
        
    def getEncodedImage(self, image):
        encoded_image = self._session.run(self._hidden_layer, feed_dict={self._input_layer:[image]})
        return encoded_image
    
    def getDecodedImage(self, image):
        decoded_image = self._session.run(self._output_layer, feed_dict={self._input_layer:[image]})
        return decoded_image



class Autoencoder(object):
    
    def __init__(self):    
        
      # Encoding
        input_layer = Input(shape=(28, 28, 1)) 
        encoding_conv_layer_1 = Conv2D(16, (3, 3), activation='relu', 
                                       padding='same')(input_layer)
        encoding_pooling_layer_1 = MaxPooling2D((2, 2),                   
                                     padding='same')(encoding_conv_layer_1)
        encoding_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', 
                                     padding='same')(encoding_pooling_layer_1)
        encoding_pooling_layer_2 = MaxPooling2D((2, 2), 
                                     padding='same')(encoding_conv_layer_2)
        encoding_conv_layer_3 = Conv2D(8, (3, 3), activation='relu', 
                                     padding='same')(encoding_pooling_layer_2)
        code_layer = MaxPooling2D((2, 2), padding='same')(encoding_conv_layer_3)
        
        # Decoding
        decodging_conv_layer_1 = Conv2D(8, (3, 3), activation='relu', 
                                      padding='same')(code_layer)
        decodging_upsampling_layer_1 = UpSampling2D((2, 2))(decodging_conv_layer_1)
        decodging_conv_layer_2 = Conv2D(8, (3, 3), activation='relu', 
                                      padding='same')(decodging_upsampling_layer_1)
        decodging_upsampling_layer_2 = UpSampling2D((2, 2))(decodging_conv_layer_2)
        decodging_conv_layer_3 = Conv2D(16, (3, 3),                      
                                   activation='relu')(decodging_upsampling_layer_2)
        decodging_upsampling_layer_3 = UpSampling2D((2, 2))(decodging_conv_layer_3)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', 
                                      padding='same')(decodging_upsampling_layer_3)
        
        self._model = Model(input_layer, output_layer)
        self._model.compile(optimizer='adadelta', loss='binary_crossentropy')
        
      
        
    def plot_loss(history):
            plt.plot(history.history['loss'], label='training loss')
            plt.plot(history.history['val_loss'], label=' validation loss')
            #plt.ylim([0, 10])
            plt.xlabel('Epopch')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True)
            plt.savefig( 'Teste'+ str(1) + '/error_datalenght.png')
            plt.show()

        
    def train(self, input_train, input_test, batch_size, epochs):    
        
        self._model.fit(input_train, 
                        input_train,
                        epochs = epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                                input_test, 
                                input_test))
        
       
       
        checkpoint_path = 'Teste'+ str(1) + '/weights-{epoch:03d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)

       # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
            save_freq=2000)
        
        
        history =self._model.fit (input_train, 
                        input_train,
                        epochs = epochs,
                        callbacks=[cp_callback],
                        batch_size=batch_size,
                        shuffle=True,
                        steps_per_epoch = 1,
                        validation_data=(
                                input_train, 
                                input_train))
        plt.plot(history.history['sparse_categorical_accuracy'])
        plt.plot(history.history['val_sparse_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid(True)
        plt.savefig('Teste_'+ str(1) + '/accuracy_datalenght.png')
        plt.show()
        tf.keras.models.save_model (self._model,'teste1' +  '/ModelTreinamento')
        
    
    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image

class Autoencoder1(object):
    def __init__(self, inout_dim, encoded_dim):
        learning_rate = 0.1 
        
        # Weights and biases
        hiddel_layer_weights = tf.Variable(tf.random.normal([inout_dim, encoded_dim]))
        hiddel_layer_biases = tf.Variable(tf.random.normal([encoded_dim]))
        output_layer_weights = tf.Variable(tf.random.normal([encoded_dim, inout_dim]))
        output_layer_biases = tf.Variable(tf.random.normal([inout_dim]))
        
        # Neural network
        self._input_layer = tf.placeholder('float', [None, inout_dim])
        self._hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(self._input_layer, hiddel_layer_weights), hiddel_layer_biases))
        self._output_layer = tf.matmul(self._hidden_layer, output_layer_weights) + output_layer_biases
        self._real_output = tf.placeholder('float', [None, inout_dim])
        
        self._meansq = tf.reduce_mean(tf.square(self._output_layer - self._real_output))
        self._optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self._meansq)
        self._training = tf.global_variables_initializer()
        self._session = tf.Session()
        
    def train(self, input_train, input_test, batch_size, epochs):
        self._session.run(self._training)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(int(input_train.shape[0]/batch_size)):
                epoch_input = input_train[ i * batch_size : (i + 1) * batch_size ]
                _, c = self._session.run([self._optimizer, self._meansq], feed_dict={self._input_layer: epoch_input, self._real_output: epoch_input})
                epoch_loss += c
                print('Epoch', epoch, '/', epochs, 'loss:',epoch_loss)
        
    def getEncodedImage(self, image):
        encoded_image = self._session.run(self._hidden_layer, feed_dict={self._input_layer:[image]})
        return encoded_image
    
    def getDecodedImage(self, image):
        decoded_image = self._session.run(self._output_layer, feed_dict={self._input_layer:[image]})
        return decoded_image
    