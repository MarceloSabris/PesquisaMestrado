import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
 
import glob
import os
import time
import cv2
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

#(x_train, _), (x_test, _) =  tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#x_train = x_train[0]
#x_test = x_test[0]


def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))
  
model1 = tf.keras.models.load_model("C:\Source\Relation-Network-Tensorflow\decoder\model", compile=False)

model1.compile(loss=SSIMLoss, optimizer='adam', metrics=SSIMLoss)

json_model = model1.to_json()
#save the model architecture to JSON file
with open('C:\\Source\\Relation-Network-Tensorflow\\decoder\\teste.json', 'w') as json_file:
    json_file.write(json_model)


def conv2dWheights (model,filter,strides1, position,name1,other  ):
 values = model.layers[position].get_weights()
 weights = tf.keras.initializers.constant(values[0])
 bias =  tf.keras.initializers.constant(values[1])
 encoded = tf.keras.layers.Conv2D(filter, kernel_size=3, strides= strides1, padding='same',  kernel_initializer = weights,bias_initializer=bias,name=name1)(other)
 return encoded

def conv2dTransposeWheights (model,filter,strides1, position,name1,other  ):
 values = model.layers[position].get_weights()
 weights = tf.keras.initializers.constant(values[0])
 bias =  tf.keras.initializers.constant(values[1])
 encoded = tf.keras.layers.Conv2DTranspose(filter, kernel_size=3, strides= strides1, padding='same',  kernel_initializer = weights,bias_initializer=bias,name=name1)(other)
 return encoded

def BatchNormalization (model,position,name1,other  ):
 values = model.layers[position].get_weights()
 beta = tf.keras.initializers.constant(model.layers[position].beta)
 gamma =  tf.keras.initializers.constant(model.layers[position].gamma)
 moving_mean = tf.keras.initializers.constant(model.layers[position].moving_mean)
 moving_variance = tf.keras.initializers.constant(model.layers[position].moving_variance)
 encoded = tf.keras.layers.BatchNormalization(name=name1,  beta_initializer= beta,gamma_initializer=gamma,moving_mean_initializer=moving_mean , moving_variance_initializer=moving_variance)(other)
 return encoded

inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer')
encoded =conv2dWheights (model1,32,1, 1,'conv_1',inputs  ) 
encoded = BatchNormalization(model1,2,'batchnorm_1',encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_1')(encoded)
# Conv Block 2 -> BatchNorm->leaky Relu
encoded = conv2dWheights(model1,64,2,4,'conv_2',encoded)
encoded = BatchNormalization(model1,5,'batchnorm_2',encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_2')(encoded)
# Conv Block 3 -> BatchNorm->leaky Relu
encoded = conv2dWheights(model1,64,2,7,'conv_3',encoded)
encoded = BatchNormalization(model1,8,'batchnorm_3',encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_3')(encoded)

#Decoder
decoded = conv2dTransposeWheights(model1,64,1,10,'conv_transpose_1', encoded  )
decoded = BatchNormalization(model1,11,'batchnorm_4',decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_4')(decoded)
# DeConv Block 2-> BatchNorm->leaky Relu

decoded =conv2dTransposeWheights(model1,64,2,13,'conv_transpose_2', decoded  )
decoded = BatchNormalization(model1,14,'batchnorm_5',decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_5')(decoded)
# DeConv Block 3-> BatchNorm->leaky Relu
decoded  =conv2dTransposeWheights(model1,32,2,16,'conv_transpose_3', decoded  )
decoded = BatchNormalization(model1,17,'batchnorm_6',decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_6')(decoded)
# output
outputs = conv2dTransposeWheights(model1,3,1,19,'conv_transpose_4', decoded  )





autoencoder2 = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
autoencoder2.compile(optimizer=optimizer, loss=SSIMLoss)
autoencoder2.summary()

i=0
for layer in autoencoder2.layers:
 if layer.trainable_weights:
    if layer.name.find("batch") == -1:
      print(layer.name)
      print(model1.layers[i].get_weights()[1] == layer.bias)
      print(model1.layers[i].get_weights()[0] == layer.get_weights()[0])
   
    

    else:
      print(layer.name)
      print(model1.layers[i].gamma == layer.gamma)
      print(model1.layers[i].beta == layer.beta)
      print(model1.layers[i].moving_mean  == layer.moving_mean )
      print(model1.layers[i].moving_variance == layer.moving_variance)
 i=i+1

decoded_imgs = autoencoder2.predict(x_test)
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

plt.show()




inputs = model1.layers[0]
# Conv Block 1 -> BatchNorm->leaky Relu
encoded =model1.layers[1]
encoded =model1.layers[2]
encoded = model1.layers[3]
# Conv Block 2 -> BatchNorm->leaky Relu
encoded = model1.layers[4]
encoded = model1.layers[5]
encoded = model1.layers[6]
# Conv Block 3 -> BatchNorm->leaky Relu
encoded = model1.layers[7]
encoded = model1.layers[8]
encoded = model1.layers[9]

#Decoder
decoded = model1.layers[10]
decoded = model1.layers[11]
decoded = model1.layers[12]
# DeConv Block 2-> BatchNorm->leaky Relu

decoded =model1.layers[13]
decoded = model1.layers[14]
decoded = model1.layers[15]
# DeConv Block 3-> BatchNorm->leaky Relu
decoded  =model1.layers[16]
decoded = model1.layers[17]
decoded = model1.layers[18]
# output
outputs = model1.layers[19]


autoencoder2 = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
autoencoder2.compile(optimizer=optimizer, loss=SSIMLoss)
autoencoder2.summary()

decoded_imgs = autoencoder2.predict(x_test)
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

plt.show()







decoded_imgs = model1.predict(x_test)
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

plt.show()




#Decoder
decoded = conv2dTransposeWheights(model1,64,1,10,'conv_transpose_1', encoded  )
decoded = BatchNormalization(model1,11,'batchnorm_4',encoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_4')(decoded)
# DeConv Block 2-> BatchNorm->leaky Relu

decoded =conv2dTransposeWheights(model1,64,2,13,'conv_transpose_2', encoded  )
decoded = BatchNormalization(model1,14,'batchnorm_5',encoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_5')(decoded)
# DeConv Block 3-> BatchNorm->leaky Relu
decoded = conv2dTransposeWheights(model1,32,2,16,'conv_transpose_3', encoded  )
decoded =BatchNormalization(model1,17,'batchnorm_6',encoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_6')(decoded)

outputs = tf.keras.layers.Conv2DTranspose(3, kernel_size= 3,strides= 1,padding='same', activation='sigmoid', name='conv_transpose_4')(decoded)

autoencoder2 = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
autoencoder2.compile(optimizer=optimizer, loss=SSIMLoss)
autoencoder2.summary()

    





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

plt.show()


for layer in model1.layers:
 if layer.trainable_weights:
   print(layer.name)





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

#(x_train, _), (x_test, _) =  tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = np.reshape(x_train, (len(x_train), 128, 128, 3))
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
#x_train = x_train[0]
#x_test = x_test[0]


inputs = tf.keras.Input(shape=(128, 128, 3), name='input_layer')
# Conv Block 1 -> BatchNorm->leaky Relu
encoded = tf.keras.layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
encoded = tf.keras.layers.BatchNormalization(name='batchnorm_1')(encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_1')(encoded)
# Conv Block 2 -> BatchNorm->leaky Relu
encoded = tf.keras.layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(encoded)
encoded = tf.keras.layers.BatchNormalization(name='batchnorm_2')(encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_2')(encoded)
# Conv Block 3 -> BatchNorm->leaky Relu
encoded = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_3')(encoded)
encoded = tf.keras.layers.BatchNormalization(name='batchnorm_3')(encoded)
encoded = tf.keras.layers.LeakyReLU(name='leaky_relu_3')(encoded)
#Decoder
# DeConv Block 1-> BatchNorm->leaky Relu
decoded = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides= 1, padding='same',name='conv_transpose_1')(encoded)
decoded = tf.keras.layers.BatchNormalization(name='batchnorm_4')(decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_4')(decoded)
# DeConv Block 2-> BatchNorm->leaky Relu
decoded = tf.keras.layers.Conv2DTranspose(64,kernel_size= 3, strides= 2, padding='same', name='conv_transpose_2')(decoded)
decoded = tf.keras.layers.BatchNormalization(name='batchnorm_5')(decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_5')(decoded)
# DeConv Block 3-> BatchNorm->leaky Relu
decoded = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides= 2, padding='same', name='conv_transpose_3')(decoded)
decoded = tf.keras.layers.BatchNormalization(name='batchnorm_6')(decoded)
decoded = tf.keras.layers.LeakyReLU(name='leaky_relu_6')(decoded)
# output
outputs = tf.keras.layers.Conv2DTranspose(3, kernel_size= 3,strides= 1,padding='same', activation='sigmoid', name='conv_transpose_4')(decoded)


autoencoder = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(lr = 0.0005)
autoencoder.compile(optimizer=optimizer, loss=SSIMLoss)
autoencoder.summary()

    





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

plt.show()

history=autoencoder.fit(x_train, x_train,
                epochs=20,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test)
                )

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig("C:\Source\Relation-Network-Tensorflow\decoder\Treino.png")
tf.keras.models.save_model (autoencoder,"C:\Source\Relation-Network-Tensorflow\decoder\model")
autoencoder.save_weights("C:\Source\Relation-Network-Tensorflow\decoder\weights")
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

plt.savefig("C:\Source\Relation-Network-Tensorflow\decoder\teste.png")
