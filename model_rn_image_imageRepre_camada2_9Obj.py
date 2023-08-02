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
from ops import conv2d, fc
from util import log
import numpy as np
from vqa_util import question2str, answer2str
class Model(object):

    def __init__(self, config,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information


        self.config = config
        self.batch_size = self.config.batch_size
        self.img_size = self.config.data_info[0]
        self.c_dim = self.config.data_info[2]
        self.q_dim = self.config.data_info[3]
        self.a_dim = self.config.data_info[4]
        self.conv_info = self.config.conv_info
             
        # create placeholders for the input
        if self.config.is_loadImage == True:
            self.img = tf.compat.v1.placeholder(
             name='img', dtype=tf.float32,
             shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
            )
        else: 
            self.img = tf.compat.v1.placeholder(
            name='img', dtype=tf.float32,
            shape=[self.batch_size, 1],
            ) 
           
         
        self.q =  tf.compat.v1.placeholder(
            name='q', dtype=tf.float32, shape=[self.batch_size, self.q_dim],
        )
        self.a =  tf.compat.v1.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )
        self.imgDecod = tf.compat.v1.placeholder(
            name='imgDecod', dtype=tf.int16,
            shape=[self.batch_size,1]
        )
        self.codImag = tf.compat.v1.placeholder(
            name='codImag', dtype=tf.float32,
            shape=[self.batch_size,2,2,9]
        )
        self.codImagOri = tf.compat.v1.placeholder(
            name='codImagOri', dtype=tf.float32,
            shape=[self.batch_size,2,2,9]
        )


        self.is_training =  tf.compat.v1.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
            #self.imgDecod : batch_chunk['imgDecod'],
            self.codImag :  batch_chunk['codImag'] ,
            self.codImagOri :  batch_chunk['codImagOri'] 
            
            
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def get_feed_dict2(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk[0],  # [B, h, w, c]
            self.q: batch_chunk[1],  # [B, n]
            self.a: batch_chunk[2],  # [B, m]
            #self.imgDecod : batch_chunk[3],
            self.codImag :  batch_chunk[4] ,
            self.codImagOri :  batch_chunk[5] 
            
            
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd



    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

            # Classification accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return tf.reduce_mean(loss), accuracy
        # }}}

        def concat_coor1(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.compat.v1.to_float(coor)], axis=1)
            return o
        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.compat.v1.to_float(coor)], axis=1)
            return o
        def g_theta(o_i, o_j, q, scope='g_theta', reuse=True):
            with tf.compat.v1.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                g_1 = fc(tf.concat([o_i, o_j, q], axis=1), 256, name='g_1')
                g_2 = fc(g_1, 256, name='g_2')
                g_3 = fc(g_2, 256, name='g_3')
                #g_4 = fc(g_3, 256, name='g_4')
                return g_3

       
   
        def BatchNormalization (model,position,name1,other  ): 
            values = model.layers[position].get_weights()
            beta = tf.keras.initializers.constant(model.layers[position].beta)
            gamma =  tf.keras.initializers.constant(model.layers[position].gamma)
            moving_mean = tf.keras.initializers.constant(model.layers[position].moving_mean)
            moving_variance = tf.keras.initializers.constant(model.layers[position].moving_variance)
            encoded = tf.keras.layers.BatchNormalization(name=name1,  beta_initializer= beta,gamma_initializer=gamma,moving_mean_initializer=moving_mean , moving_variance_initializer=moving_variance)(other)
            return encoded        
            
        def SSIMLoss(y_true, y_pred):
           return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred,1.0))
        optimizer = tf.keras.optimizers.Adam(lr = 0.017)

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, q,repImg, scope='CONV', is_loadImage=False): 
            with  tf.compat.v1.variable_scope(scope) as scope:
                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                if is_loadImage == True: 
                  conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
                  conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
                  conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                  conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')
                  d = conv_4.get_shape().as_list()[1]
                else:
                  d = repImg.get_shape().as_list()[1]
                
                all_g = []
                for i in range(d*d):
                    #o_i = repImg[:, int(i / d), int(i % d), :]
                    if is_loadImage == True:
                       o_i = conv_4[:, int(i / d), int(i % d), :]
                       o_i = concat_coor(o_i, i, d)
                    else: 
                       o_i = repImg[:, int(i / d), int(i % d), :] 
                    for j in range(d*d):
                        if is_loadImage == True:
                           o_j = conv_4[:, int(i / d), int(i % d), :]
                           o_j = concat_coor(o_i, i, d)
                        else: 
                           o_j = repImg[:, int(i / d), int(i % d), :]
                        
                        # o_j = repImg[:, int(j / d), int(j % d), :]
                        #o_j = conv_4[:, int(i / d), int(i % d), :]
                        o_j = concat_coor(o_j, j, d)
                        if i == 0 and j == 0:
                            g_i_j = g_theta(o_i, o_j, q, reuse=False)
                        else:
                            g_i_j = g_theta(o_i, o_j, q, reuse=True)
                        all_g.append(g_i_j)

                all_g = tf.stack(all_g, axis=0)
                all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
                return all_g
        def execute_op_as_image(session,op):
            """
            Evaluate the given `op` and return the content PNG image as `PIL.Image`.

            - If op is a plot op (e.g. RGBA Tensor) the image or
            a list of images will be returned
            - If op is summary proto (i.e. `op` was a summary op),
            the image content will be extracted from the proto object.
            """
            print ("Executing: " + str(op))
            ret = session.run(op)
            tfplot.close()

            if isinstance(ret, np.ndarray):
                if len(ret.shape) == 3:
                    # single image
                    return Image.fromarray(ret)
                elif len(ret.shape) == 4:
                    return [Image.fromarray(r) for r in ret]
                else:
                    raise ValueError("Invalid rank : %d" % len(ret.shape))

            elif isinstance(ret, (str, bytes)):
                from io import BytesIO
                s = tf.Summary()
                s.ParseFromString(ret)
                ims = []
                for i in range(len(s.value)):
                    png_string = s.value[i].image.encoded_image_string
                    im = Image.open(BytesIO(png_string))
                    ims.append(im)
                tfplot.close()
                if len(ims) == 1: return ims[0]
                else: return ims

            else:
                raise TypeError("Unknown type: " + str(ret))
        def f_phi(g, scope='f_phi'):
            with tf.compat.v1.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                #fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_1, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, n, activation_fn=None, name='fc_3')
                return fc_3

        g = CONV(self.img, self.q, self.codImagOri,scope='CONV',is_loadImage=self.config.is_loadImage)
        logits = f_phi(g, scope='f_phi')
        self.all_preds = tf.nn.softmax(logits)
        self.loss, self.accuracy = build_loss(logits, self.a)
        self.testAcuracy = 0 
        # Add summaries'
        def draw_iqa(img, q, target_a, pred_a):
            fig, ax = tfplot.subplots(figsize=(6, 6))
            ax.imshow(img)
            ax.set_title(question2str(q))
            ax.set_xlabel(answer2str(target_a)+answer2str(pred_a, 'Predicted'))
            return fig

        #draw_iqa(self.img, self.q, self.a, self.all_preds)


        #op = tfplot.summary.plot_many('IQA/',
         #                            draw_iqa, [self.img, self.q, self.a, self.all_preds],
          #                           max_outputs=4)
        #execute_op_as_image(tf.compat.v1.Session(),op) 
                
        try:
            tfplot.summary.plot_many('IQA/',
                                     draw_iqa, [self.img, self.q, self.a, self.all_preds],
                                     max_outputs=4,
                                     collections=["plot_summaries"])
        except:
            pass


        
        tf.compat.v1.summary.scalar("loss/accuracy", self.accuracy)
        tf.compat.v1.summary.scalar("loss/cross_entropy", self.loss)
        log.warn('Successfully loaded the model.')
    
   