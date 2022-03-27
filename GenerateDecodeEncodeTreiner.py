import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
import glob
import os
import time

import numpy as np

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
  img, q, a,d,g,g1 = dataset.get_data(id)
  return (id, img.astype(np.float32), q.astype(np.float32), a.astype(np.float32))


for id in range(100):
  train_imgs.append(load_fn(dataset_train.ids[id],dataset_train)[1])

data_id = dataset_test.ids

for id in range(100):
  test_imgs.append(load_fn(dataset_test.ids[id],dataset_test)[1])



       
a="" 
batch_train = "" 
b=""
batch_test=""
x_test = np.array(test_imgs)
x_train =  np.array(train_imgs)

#(x_train, _), (x_test, _) =  tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#x_train = x_train[0]
#x_test = x_test[0]




inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer')
# Conv Block 1 -> BatchNorm->leaky Relu
encoded = tf.keras.layers.Conv2D(30, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
encoded = tf.keras.layers.BatchNormalization(name='batchnorm_1')(encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_1')(encoded)

#tf.nn.relu  X tf.keras.layers.LeakyReLU

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

#Decoder
# DeConv Block 1-> BatchNorm->leaky Relu
decoded = tf.keras.layers.Conv2DTranspose(30, kernel_size=3, strides= 1, padding='same',name='conv_transpose_1')(encoded)
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

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))

autoencoder = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(lr = 0.017)
autoencoder.compile(optimizer=optimizer, loss=SSIMLoss)
autoencoder.summary()
autoencoder.load_weights("C:\source\PesquisaMestrado\decoder1\weights4")

#history=autoencoder.fit(x_train, x_train,
#                epochs=20,
#                batch_size=170,
#                shuffle=True,
#                validation_data=(x_test, x_test)
#                )

#plt.plot(history.history['loss'], label='loss')
#plt.plot(history.history['val_loss'], label = 'val_loss')
#plt.xlabel('Epoch')
#plt.ylabel('loss')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')
#plt.savefig("C:\Source\Relation-Network-Tensorflow\decoder\Treino3.png")
tf.keras.models.save_model (autoencoder,"C:\source\PesquisaMestrado\decoder1\model3")
autoencoder.save_weights("C:\source\PesquisaMestrado\decoder1\weights4")
#weitghts = encoded.get_weights()


decoded_imgs = autoencoder.predict(x_test)
n = 10
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

plt.savefig("C:\source\PesquisaMestrado\decoder1\Treino13.png")
