import glob
import os
import time
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

batch_size = 128
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.astype('float32')
x_test = x_test / 255.
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).\
shuffle(60000).batch(128)

plt.figure(figsize=(10, 10))
for images in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i,:,:,0].numpy().astype("uint8"), cmap='gray')
        plt.axis("off")
normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./255)
plt.show()


normalized_ds = train_dataset.map(lambda x: normalization_layer(x))
image_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image)) 

input_encoder = (28, 28, 1)
input_decoder = (2,)

def encoder(input_encoder):
    
    inputs = keras.Input(shape=input_encoder, name='input_layer')
    x = layers.Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2D(64, 3, 2, padding='same', name='conv_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    #x = layers.Dropout(rate = 0.25)(x)

    x = layers.Conv2D(64, 3, 1, padding='same', name='conv_4')(x)
    x = layers.BatchNormalization(name='bn_4')(x)
    x = layers.LeakyReLU(name='lrelu_4')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    flatten = layers.Flatten()(x)
    bottleneck = layers.Dense(2, name='dense_1')(flatten)
    model = tf.keras.Model(inputs, bottleneck, name="Encoder")
    return model
enc = encoder(input_encoder)

enc.summary()

def decoder(input_decoder):
    
    inputs = keras.Input(shape=input_decoder, name='input_layer')
    x = layers.Dense(3136, name='dense_1')(inputs)
    #x = tf.reshape(x, [-1, 7, 7, 64], name='Reshape_Layer')
    x = layers.Reshape((7,7,64), name='Reshape_Layer')(x)
    x = layers.Conv2DTranspose(64, 3, strides= 1, padding='same',name='conv_transpose_1')(x)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.LeakyReLU(name='lrelu_1')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2DTranspose(64, 3, strides= 2, padding='same', name='conv_transpose_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.LeakyReLU(name='lrelu_2')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    x = layers.Conv2DTranspose(32, 3, 2, padding='same', name='conv_transpose_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.LeakyReLU(name='lrelu_3')(x)
    #x = layers.Dropout(rate = 0.25)(x)
    
    outputs = layers.Conv2DTranspose(1, 3, 1,padding='same', activation='sigmoid', name='conv_transpose_4')(x)
    model = tf.keras.Model(inputs, outputs, name="Decoder")
    return model

dec = decoder(input_decoder)

dec.summary()

dec.save('ae-dec-fashion.h5')
enc.save('ae-enc-fashion.h5')

dec.summary()

#model.layers[1].get_weights()
#model.save('autoencoder.h5')

optimizer = tf.keras.optimizers.Adam(lr = 0.0005)

from tensorflow.keras import backend as K

def ae_loss(y_true, y_pred):
    loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):

    with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
      
        latent = enc(images, training=True)
        generated_images = dec(latent, training=True)
        loss = ae_loss(images, generated_images)
        
    gradients_of_enc = encoder.gradient(loss, enc.trainable_variables)
    gradients_of_dec = decoder.gradient(loss, dec.trainable_variables)
    
    
    optimizer.apply_gradients(zip(gradients_of_enc, enc.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_dec, dec.trainable_variables))
    return loss

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    latent = enc(test_input, training=False)
    predictions = dec(latent, training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        plt.axis('off')

    plt.savefig('tf_ae/fashion/images/image_at_epoch_{:d}.png'.format(epoch))
    plt.show()

def train(dataset, epochs):
    #for epoch in range(epochs):
    start = time.time()
    i = 0
    loss_ = []
    for image_batch in dataset:
        i += 1
        loss = train_step(image_batch)
        #loss_.append(loss)

    #print("Loss",np.mean(loss_))    
    seed = image_batch[:25]
    #display.clear_output(wait=True)
    generate_and_save_images([enc,dec],
                              1 + 1, #epoch
                              seed)
    # Save the model every 15 epochs
    #if (epoch + 1) % 15 == 0:
    #checkpoint.save(file_prefix = checkpoint_prefix)
    enc.save_weights('training_weights/enc_'+ str(1)+'.h5')
    dec.save_weights('training_weights/dec_'+ str(1)+'.h5')
    print ('Time for epoch {} is {} sec'.format(1 + 1, time.time()-start))

    # Generate after the final epoch
    #display.clear_output(wait=True)
    generate_and_save_images([enc,dec],
                            epochs,
                            seed)

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    latent = enc(test_input, training=False)
    predictions = dec(latent, training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray')
        plt.axis('off')


    plt.savefig('image_at_epoch_{:d}.png'.format(epoch))
    plt.show()

train(normalized_ds, 40)
enc.load_weights('training_weights/enc_1.h5')
dec.load_weights('training_weights/dec_1.h5')

embeddings = None

n_to_show = 5000
figsize = 10

index = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[index]

embeddings = enc.predict(example_images)


plt.figure(figsize=(figsize, figsize))
plt.scatter(embeddings[:, 0] , embeddings[:, 1], alpha=0.5, s=2)
plt.xlabel("Dimension-1", size=20)
plt.ylabel("Dimension-2", size=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.title("Projection of 2D Latent-Space (Fashion-MNIST)", size=20)
plt.show()


min_x = min(embeddings[:, 0])
max_x = max(embeddings[:, 0])
min_y = min(embeddings[:, 1])
max_y = max(embeddings[:, 1])
# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}
figsize = 15

latent = enc.predict(x_test[:25])
reconst = dec.predict(latent)

fig = plt.figure(figsize=(figsize, 10))
#fig.subplots_adjust(wspace=-0.021)

for i in range(25):
    ax = fig.add_subplot(5, 5, i+1)
    ax.axis('off')
    plt.text(0.5, -0.15, str(label_dict[y_test[i]]), fontsize=15, ha='center', transform=ax.transAxes)
    #plt.subplots_adjust(wspace=None, hspace=None)
    plt.imshow(reconst[i, :,:,0]*255, cmap = 'gray')
plt.show()    

min_x = min(embeddings[:, 0])
max_x = max(embeddings[:, 0])
min_y = min(embeddings[:, 1])
max_y = max(embeddings[:, 1])

x = np.random.uniform(low=min_x,high=max_x, size = (10,1))
y = np.random.uniform(low=min_y,high=max_y, size = (10,1))
bottleneck = np.concatenate((x, y), axis=1)
reconst = dec.predict(bottleneck)

fig = plt.figure(figsize=(15, 10))

for i in range(10):
    ax = fig.add_subplot(5, 5, i+1)
    ax.axis('off')
    plt.text(0.5, -0.15, str(np.round(bottleneck[i],1)), fontsize=10, ha='center', transform=ax.transAxes)
    
    plt.imshow(reconst[i, :,:,0]*255, cmap = 'gray')

    
plt.show()    