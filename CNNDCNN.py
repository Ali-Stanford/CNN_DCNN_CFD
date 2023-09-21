##### Point-cloud deep learning for prediction of fluid flow fields on irregular geometries (supervised learning) #####

#Author: Ali Kashefi (kashefi@stanford.edu)
#Description: Implementation of CNN-DCNN for *supervised learning* of computational mechanics
#Version: 1.0
#Guidance: We recommend opening and running the code on **[Google Colab](https://research.google.com/colaboratory)** as a first try. 

import numpy as np
import tensorflow as tf
from tensorflow import keras

class CNN_DCNN(tf.keras.Model):
    def __init__(self, input_shape):
        super(CNN_DCNN, self).__init__()

        # CNN Layers
        self.conv_layers = [
            keras.layers.Conv2D(16, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False, kernel_initializer='random_normal', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2),

            keras.layers.Conv2D(1024, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(alpha=0.2)
        ]

        # DCNN Layers
        self.deconv_layers = [
           keras.layers.Conv2DTranspose(1024, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu'),

            keras.layers.Conv2DTranspose(1, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer='random_normal', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('relu')
        ]

    def call(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        for layer in self.deconv_layers:
            x = layer(x)
        return x

#Usage Example
#nx = 128 # x resolution (e.g., 128x128 or 512x512)
#ny = 128 # y resolution (e.g., 128x128 or 512x512)

#model = CNN_DCNN(input_shape=(nx, ny, 1))
#model.build(input_shape=(None, nx, ny, 1))
#model.summary()

# Generate some fake data to test the model
#n_training = 100
#n_validation = 10
#input_training = np.random.rand(n_training, nx, ny, 1).astype(np.float32)
#output_training = np.random.rand(n_training, nx, ny, 1).astype(np.float32)
#input_validation = np.random.rand(n_validation, nx, ny, 1).astype(np.float32)
#output_validation = np.random.rand(n_validation, nx, ny, 1).astype(np.float32)

#model.compile(keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.0,)
#                   , loss='mean_squared_error', metrics=['mean_squared_error'])

#results = model.fit(input_training, output_training, batch_size=256, epochs=1000, verbose=1, validation_data=(input_validation, output_validation))
