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


model = tf.keras.models.load_model("C:\Source\Relation-Network-Tensorflow\decoder\model8", compile=False) 
                

model.compile()
model.summary()
model.load_weights("C:\Source\Relation-Network-Tensorflow\decoder\weights8")