from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils

K.set_image_dim_ordering('th')

# load input data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print ("X_train.shape = " + str(X_train.shape))
print ("Y_train.shape = " + str(Y_train.shape))
print ("X_test.shape = " + str(X_test.shape))
print ("Y_test.shape = " + str(Y_test.shape))

# Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Preprocess class labels
Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

print ("X_train.shape = " + str(X_train.shape))
print ("Y_train.shape = " + str(Y_train.shape))
print ("X_test.shape = " + str(X_test.shape))
print ("Y_test.shape = " + str(Y_test.shape))


input_shape = (1, 28, 28)

X_input = Input(input_shape)

X = ZeroPadding2D((3, 3))(X_input)

X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
X = BatchNormalization(axis = 3, name = 'bn0')(X)
X = Activation('relu')(X)

X = MaxPooling2D((2, 2), name = 'max_pool')(X)

X = Flatten()(X)
X = Dense(10, activation='softmax', name='fc')(X)

model = Model(inputs = X_input, outputs = X, name='mnistmodel')
model.compile(optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss = "mean_squared_error", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 5, batch_size = 32)

preds = model.evaluate(x = X_test, y = Y_test)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))