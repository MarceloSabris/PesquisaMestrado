from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import tf_slim  as slim
try:
    import tfplot
except:
    pass
from PIL import Image, ImageDraw


import numpy as np
def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))

                

optimizer = tf.keras.optimizers.Adam(lr = 0.017)


model1 = tf.keras.models.load_model("C:\Source\PesquisaMestrado\decoder\model8", compile=False) 
                

model1.compile()
model1.summary()
model1.load_weights("C:\Source\PesquisaMestrado\decoder\weights8")

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
encoded =conv2dWheights (model1,43,1, 1,'conv_1',inputs  ) 
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