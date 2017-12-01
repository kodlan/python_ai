from keras import backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
import tensorflow as tf
from keras.utils import plot_model
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib

sess = tf.Session()
K.set_session(sess)

K.set_image_dim_ordering('th')

# Use this only for export of the model.
# This must come before the instantiation of model
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)

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

X = MaxPooling2D((2, 2), name = 'max_pool0')(X)

X = Conv2D(15, (3, 3), strides = (1, 1), name = 'conv1')(X)
X = BatchNormalization(axis = 3, name = 'bn1')(X)
X = Activation('relu')(X)

X = MaxPooling2D((2, 2), name = 'max_pool1')(X)

X = Dropout(0.2)(X)
X = Flatten()(X)
X = Dense(128, activation='relu')(X)
X = Dense(64, activation='relu')(X)
X = Dense(10, activation='softmax', name='fc')(X)

model = Model(inputs = X_input, outputs = X, name='mnistmodel')
model.compile(optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), loss = "mean_squared_error", metrics = ["accuracy"])
model.fit(x = X_train, y = Y_train, epochs = 10, batch_size = 128)

preds = model.evaluate(x = X_test, y = Y_test)

print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))

print("model.output = ")
print(model.output)
print("model.input = ")
print(model.input)

plot_model(model, to_file='model.png')

# this is not working for some reason
# # Create a builder
# builder = tf.saved_model.builder.SavedModelBuilder('./SavedModel/')
# builder.add_meta_graph_and_variables(sess,
#                                        [tf.saved_model.tag_constants.SERVING],
#                                        signature_def_map=None,
#                                        assets_collection=None)
# builder.save()

model_work_dir = './model/'

# Save graph
saver = tf.train.Saver()
tf.train.write_graph(sess.graph_def, model_work_dir, 'model.pbtxt')
model_ckpt_name = saver.save(sess, model_work_dir + 'model.ckpt')

# Freeze the graph
input_graph_path = model_work_dir + 'model.pbtxt'
output_frozen_graph_name = model_work_dir + 'frozen_model.pb'
output_optimized_graph_name = model_work_dir + 'optimized_model.pb'
checkpoint_path = model_ckpt_name

output_node_names = "fc/Softmax"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
prefix_output_node_names_of_final_network = 'output_node'

freeze_graph.freeze_graph(input_graph_path, "",
                          False, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, True, "")

# Optimize graph
input_graph_def = tf.GraphDef()
with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input_1"], # an array of the input node(s)
        ["fc/Softmax"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph
f = tf.gfile.FastGFile(output_optimized_graph_name, "w")
f.write(output_graph_def.SerializeToString())