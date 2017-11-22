from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

input_folder = 'model_output/'
input_model = 'saved_model.pb'
input_file_path = osp.join(input_folder, input_model)

num_output = 10

prefix_output_node_names_of_final_network = 'output_node'
output_file_name = 'constant_graph_weights.pb'


K.set_learning_phase(0)
net_model = load_model(input_file_path)


pred = [None] * num_output
pred_node_names = [None] * num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)


sess = K.get_session()
f = 'only_the_graph_def.pb.ascii'
tf.train.write_graph(sess.graph.as_graph_def(), input_folder, f, as_text=True)
print('saved the graph definition in ascii format at: ', osp.join(input_folder, f))


constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, input_folder, output_file_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', osp.join(input_folder, output_file_name))
