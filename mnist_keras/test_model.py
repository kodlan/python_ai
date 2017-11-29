import tensorflow as tf
import numpy as np
import PIL.Image as Image

model_path = "./model/"
test_image = "./model/1.jpg"
input_node_name = 'input_1'
output_node_name = 'fc/Softmax'

img = Image.open(test_image).convert('L')
img.load()
data = np.asarray(img, dtype="int32")
data = np.reshape(data, (1, 1, 28, 28))

with tf.Session() as sess:
    tf.saved_model.loader.load(sess, [[tf.saved_model.tag_constants.TRAINING]], export_dir=model_path)
    output = sess.graph.get_tensor_by_name(output_node_name)
    sess.run(output, {input_node_name: [data]})
