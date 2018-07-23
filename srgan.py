#!/usr/bin/env python3
from keras.layers import Input, Dense, Flatten, Lambda
from keras.layers import Add, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam

import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np
import os


class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.lp1_weight = 2e-1
        self.lp2_weight = 2e-2

        # Number of residual blocks in the generator
        self.n_residual_blocks = 10

        optimizer = Adam(0.0001, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.lp = self.build_lp()
        self.lp.trainable = False
        self.lp.compile(loss=['mse', 'mse'],
                         optimizer=optimizer,
                         metrics=['accuracy'],
                         loss_weights=[self.lp1_weight, self.lp2_weight])

        # Configure data loader
        self.dataset_name = 'BSDS200'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.lp(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features[0], fake_features[1]])
        self.combined.compile(loss=['binary_crossentropy', 'mse', 'mse'],
                              loss_weights=[1e-3, self.lp1_weight, self.lp2_weight],
                              optimizer=optimizer)

    def build_lp(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        vgg1 = Model(inputs=vgg.input, outputs=vgg.layers[6].output)
        vgg2 = Model(inputs=vgg.input, outputs=vgg.layers[21].output)

        img = Input(shape=self.hr_shape)

        # Extract image features
        lp1 = vgg1(img)

        lp2 = vgg2(img)

        return Model(img, [lp1, lp2])

    def build_generator(self):

        def residual_block(layer_input):
            """Residual block described in paper"""
            d = Conv2D(64, kernel_size=3, activation='relu',
                       padding='same')(layer_input)
            d = Conv2D(64, kernel_size=3, padding='same')(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(64, kernel_size=3, activation='relu', padding='same')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)
        from keras.backend import tf
        bic = Lambda(lambda image: tf.image.resize_bicubic(
            image, [4 * self.lr_height, 4 * self.lr_width], align_corners=True))(img_lr)
        # Pre-residual block
        c1 = Conv2D(64, kernel_size=3, activation='relu',
                    padding='same')(img_lr)

        # Propogate through residual blocks
        r = residual_block(c1)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r)

        # Upsampling
        u1 = deconv2d(r)
        u2 = deconv2d(u1)

        c2 = Conv2D(64, kernel_size=3, activation='relu', padding='same')(u2)

        c3 = Conv2D(3, kernel_size=3, padding='same')(c2)

        # Generate high resolution output
        gen_hr = Add()([bic, c3])

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        # Input img
        d0 = Input(shape=self.hr_shape)

        layer_filters = [32, 64, 128, 256, 512]
        x = d0
        for filters in layer_filters:
            x = Conv2D(filters, kernel_size=3, padding='same')(x)
            x = LeakyReLU()(x)
            x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
            x = LeakyReLU()(x)

        flat = Flatten()(x)
        d1 = Dense(1024)(flat)
        d2 = LeakyReLU()(d1)
        validity = Dense(1, activation='sigmoid')(d2)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size, 1))

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.lp.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch(
                [imgs_lr, imgs_hr], [valid, image_features[0], image_features[1]])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(
            batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' %
                        (self.dataset_name, epoch, i))
            plt.close()


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=30000, batch_size=1, sample_interval=10)
