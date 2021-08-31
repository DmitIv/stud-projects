import io
import random
import warnings

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

import cv2

from keras.callbacks import TensorBoard
from keras.callbacks import Callback


class ImagesTensorBoard(TensorBoard):

    def __init__(self, *args, generator=None, **kwargs):
        super(ImagesTensorBoard, self).__init__(*args, **kwargs)

        self.generator = generator
        
        if self.generator:
            random_indx = random.randint(0, len(generator) - 1)
            randon_batch = self.generator[random_indx]
            random_imgs = randon_batch[0]
            
#             self.test_images = random.sample(list(random_imgs), 3)
            self.test_images = np.array(random_imgs)

    def on_epoch_end(self, epoch, logs=None):
        super(ImagesTensorBoard, self).on_epoch_end(epoch, logs)

        if self.generator:
        
            pred_images = self.model.predict(self.test_images)

            print("Epoch: {}, train image:".format(epoch))

            fig = self.get_fig(self.test_images, pred_images)

            tf.summary.image("Test data", self.plot_to_image(fig), step=0)

        self.writer.flush()

    def get_fig(self, test_images, pred_images, r=2, c=3):
        
        fig, axs = plt.subplots(r, c)

        for i in range(c):
            axs[0, i].imshow(test_images[i, :, :, 0], cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(pred_images[i, :, :, 0], cmap='gray')
            axs[1, i].axis('off')
                
        return fig

    def plot_to_image(self, figure):
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_acc', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

