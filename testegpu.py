import plaidml.keras
plaidml.keras.install_backend()

import tensorflow as tf
tf.test.is_gpu_available()


tf.config.experimental.list_physical_devices('GPU')


#import plaidml.keras
#plaidml.keras.install_backend()
gpu_config = tf.GPUOptions()
gpu_config.visible_device_list = "0"

session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_config))

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=3)

#work

#import numpy as np
#import os
#import time


#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
#os.environ["RUNFILES_DIR"] = "C:\Python39\share\plaidml"
#os.environ["PLAIDML_NATIVE_PATH"] = "C:\Python39\Lib\site-packages\plaidml"
#import keras
# import keras.applications as kapp
# from keras.datasets import cifar10
# (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
#batch_size = 8
#x_train = x_train[:batch_size]
#x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
#model = kapp.VGG19()
#model.compile(optimizer='sgd', loss='categorical_crossentropy',
#              metrics=['accuracy'])

#print("Running initial batch (compiling tile program)")
#y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
#print("Timing inference...")
#start = time.time()
#for i in range(10):
 #   y = model.predict(x=x_train, batch_size=batch_size)
#print("Ran in {} seconds".format(time.time() - start))
