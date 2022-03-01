import tensorflow as tf
import tf_slim as slim


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def conv2d(input, output_shape, is_train, activation_fn=tf.nn.relu,
           k_h=5, k_w=5, s_h=2, s_w=2, stddev=0.02, name="conv2d"):
    with  tf.compat.v1.variable_scope(name):
        w = tf.compat.v1.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_shape],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input, w, strides=[1, s_h, s_w, 1], padding='SAME')
        biases = tf.compat.v1.get_variable('biases', [output_shape],
                                 initializer=tf.constant_initializer(0.0))
        activation = activation_fn(conv + biases)
        bn = tf.compat.v1.layers.batch_normalization(activation, center=True, scale=True)
    return bn



def fc(input, output_shape, activation_fn=tf.nn.relu, name="fc"):
    output = slim.fully_connected(input, int(output_shape), activation_fn=activation_fn)
    return output

