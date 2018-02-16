import cv2
import numpy as np
import glob
import os

import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Flatten, Dense, Activation, Reshape

from tools import draw_box, yolo_net_out_to_car_boxes

keras.backend.set_image_dim_ordering('th')


def load_weights(model, yolo_weight_file):
    tiny_data = np.fromfile(yolo_weight_file, np.float32)[4:]

    index = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            filter_shape, bias_shape = [w.shape for w in weights]
            if len(filter_shape) > 2:  # For convolutional layers
                filter_shape_i = filter_shape[::-1]
                bias_weight = tiny_data[index:index + np.prod(bias_shape)].reshape(bias_shape)
                index += np.prod(bias_shape)
                filter_weight = tiny_data[index:index + np.prod(filter_shape_i)].reshape(filter_shape_i)
                filter_weight = np.transpose(filter_weight, (2, 3, 1, 0))
                index += np.prod(filter_shape)
                layer.set_weights([filter_weight, bias_weight])
            else:  # For regular hidden layers
                bias_weight = tiny_data[index:index + np.prod(bias_shape)].reshape(bias_shape)
                index += np.prod(bias_shape)
                filter_weight = tiny_data[index:index + np.prod(filter_shape)].reshape(filter_shape)
                index += np.prod(filter_shape)
                layer.set_weights([filter_weight, bias_weight])


model = Sequential()
model.add(Convolution2D(16, 3, 3, input_shape=(3, 448, 448), border_mode='same', subsample=(1, 1)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(256))
model.add(Dense(4096))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1470))

# NOTE: place tiny yolo weight file in the same directory
load_weights(model, './yolo-tiny.weights')

# imagePath = os.path.dirname(os.path.realpath(__file__)) + '/test_images/test1.jpg'
# image = plt.imread(imagePath)
# image_crop = image[300:650,500:,:]
# resized = cv2.resize(image_crop,(448,448))
#
#
# batch = np.transpose(resized,(2,0,1))
# batch = 2*(batch/255.) - 1
# batch = np.expand_dims(batch, axis=0)
# out = model.predict(batch)
#
# boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
# print (boxes)
#
# f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
# ax1.imshow(image)
# ax2.imshow(draw_box(boxes, plt.imread(imagePath),[[500,1280],[300,650]]))

images = [plt.imread(file) for file in glob.glob(os.path.dirname(os.path.realpath(__file__)) + '/test_images/*.jpg')]

batch = np.array([np.transpose(cv2.resize(image[300:650, 500:, :], (448, 448)), (2, 0, 1))
                  for image in images])
batch = 2 * (batch / 255.) - 1

out = model.predict(batch)
print(out)

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(11, 10))
for i, ax in zip(range(len(batch)), [ax1, ax2, ax3, ax4, ax5, ax6]):
    boxes = yolo_net_out_to_car_boxes(out[i], threshold=0.17)
    print(boxes)
    ax.imshow(draw_box(boxes, images[i], [[500, 1280], [300, 650]]))
plt.show()
