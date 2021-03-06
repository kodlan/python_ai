import os
import PIL
from PIL import Image
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, np
import matplotlib.pyplot as plt

from yolo.tools import yolo_net_out_to_car_boxes, draw_box

input_size = 416

input_image = Input(shape=(input_size, input_size, 3))

# Layer 1
x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
x = BatchNormalization(name='norm_1')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 2 - 5
for i in range(0, 4):
    x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i + 2))(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

# Layer 6
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
x = BatchNormalization(name='norm_6')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

# Layer 7 - 8
for i in range(0, 2):
    x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 7), use_bias=False)(x)
    x = BatchNormalization(name='norm_' + str(i + 7))(x)
    x = LeakyReLU(alpha=0.1)(x)

model = Model(input_image, x)
model.load_weights('tiny_yolo_custom.h5')

imagePath = os.path.dirname(os.path.realpath(__file__)) + '/test_images/test1.jpg'
# image = plt.imread(imagePath)
# resized = cv2.resize(image_crop,(input_size, input_size))
img = Image.open(imagePath)
resized = img.resize((input_size, input_size), PIL.Image.ANTIALIAS)
resized = np.array(resized)

batch = np.transpose(resized,(2,0,1))
batch = 2*(batch/255.) - 1
batch = np.expand_dims(batch, axis=0)
out = model.predict(batch)

boxes = yolo_net_out_to_car_boxes(out[0], threshold = 0.17)
print (boxes)

f,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
ax1.imshow(img)
ax2.imshow(draw_box(boxes, plt.imread(imagePath),[[500,1280],[300,650]]))