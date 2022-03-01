from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

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
        
    def train(self, input_train, input_test, batch_size, epochs):   
        self._model.fit(input_train, 
                        input_train,
                        epochs = epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                                input_test, 
                                input_test)) 
        self._model.fit(input_train, 
                        input_train,
                        epochs = epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(
                                input_test, 
                                input_test))
        
    
    def getDecodedImage(self, encoded_imgs):
        decoded_image = self._model.predict(encoded_imgs)
        return decoded_image



import numpy as np
from tensorflow.keras.datasets import fashion_mnist

import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = fashion_mnist.load_data()
# Prepare input
X_train = x_train.astype('float32') / 255.
X_test = x_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

autoencoder = Autoencoder()
autoencoder.train(X_train, X_test, 256, 50)
decoded_imgs = autoencoder.getDecodedImage(X_test)

plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    subplot = plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)

    # Reconstruction
    subplot = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    subplot.get_xaxis().set_visible(False)
    subplot.get_yaxis().set_visible(False)
plt.show()
print('teste')