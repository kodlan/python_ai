import tensorflow as tf
import numpy as np
import PIL.Image as Image

model_path = "./SavedModel/"
input_tensor_name = 'input_1:0'
output_node_name = 'fc/Softmax:0'


def load_img(img_path):
    img = Image.open(img_path).convert('L')
    img.load()
    data = np.asarray(img, dtype="float32")
    data = np.reshape(data, (1, 28, 28))
    data /= 255
    return data


float_formatter = lambda x: "%.1f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

img_data = []
# img_data.append(load_img("./model/1.jpg"))
img_data.append(load_img("./model/2.jpg"))
img_data.append(load_img("./model/3.jpg"))
img_data.append(load_img("./model/4.jpg"))
img_data.append(load_img("./model/5.jpg"))
img_data.append(load_img("./model/6.jpg"))
img_data.append(load_img("./model/7.jpg"))
img_data.append(load_img("./model/8.jpg"))
img_data.append(load_img("./model/9.jpg"))

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir=model_path)
    output = sess.graph.get_tensor_by_name(output_node_name)
    predictions = sess.run(output, {input_tensor_name: img_data})
    print(predictions)
