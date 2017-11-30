import tensorflow as tf
import numpy as np
import PIL.Image as Image

model_path = "./model/"
model_meta = "model.ckpt.meta"
image_path = "./images/"
input_tensor_name = 'input_1:0'
output_node_name = 'fc/Softmax:0'


def load_img(img_path):
    img = Image.open(img_path).convert('L')
    img.load()
    data = np.asarray(img, dtype="float32")
    data = np.reshape(data, (1, 28, 28))
    data /= 255
    data = 1 - data
    return data


float_formatter = lambda x: "%.1f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

img_data = []
img_data.append(load_img(image_path + "1.jpg"))
img_data.append(load_img(image_path + "2.jpg"))
img_data.append(load_img(image_path + "3.jpg"))
img_data.append(load_img(image_path + "4.jpg"))
img_data.append(load_img(image_path + "5.jpg"))
img_data.append(load_img(image_path + "6.jpg"))
img_data.append(load_img(image_path + "7.jpg"))
img_data.append(load_img(image_path + "8.jpg"))
img_data.append(load_img(image_path + "9.jpg"))

with tf.Session(graph=tf.Graph()) as sess:
    # this is not working for some reason
    # tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir=model_path)

    saver = tf.train.import_meta_graph(model_path + model_meta)
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    output = sess.graph.get_tensor_by_name(output_node_name)
    predictions = sess.run(output, {input_tensor_name: img_data})
    print(predictions)
