import os.path
from os import listdir

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from nearpy import Engine

BATCH_SIZE = 32
VECTOR_DIMENSION = 2048


def get_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return x


def predict(model, x):
    x_preprocessed = preprocess_input(x)
    return model.predict(x_preprocessed)


def get_image_list(image_folder):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, image_folder)
    return [dir + f for f in listdir(dir) if f.endswith(".jpg")]


def generate_vectors(model, image_list, near_db_engine):
    # counting only complete batches for now
    for batch in range(len(image_list) / BATCH_SIZE):
        print ("Generating predictions for batch #" + str(batch))

        input_batch = load_image_batch(batch, image_list, BATCH_SIZE)
        input_array = np.array(input_batch)

        # this will return values from last hidden layer of the network
        # shape = (32, 1, 1, 2048)
        output = predict(model, input_array)
        output = np.reshape(output, (BATCH_SIZE, VECTOR_DIMENSION))
        store_vectors(output, input_batch, near_db_engine)


def load_image_batch(batch, image_list, batch_size):
    input_batch = []
    for i in range(batch_size):
        input_batch.append(get_image(image_list[batch_size * batch + i]))
    return input_batch


def generate_test_vectors(model, image_list, engine):
    # ignoring batches, assuming that test images count is not that big
    img_count = len(image_list)

    input_batch = load_image_batch(0, image_list, img_count)
    input_array = np.array(input_batch)

    output = predict(model, input_array)
    output = np.reshape(output, (img_count, VECTOR_DIMENSION))

    search_db_closest_match(output, engine)


def search_db_closest_match(feature_vectors, engine):
    for i in range(feature_vectors.shape[0]):
        print ("Feature shape = " + str(feature_vectors[i].flatten().shape))
        result = engine.neighbours(feature_vectors[i].flatten())
        print result


def store_vectors(output, input_batch, near_db_engine):
    for i in range(len(input_batch) - 1):
        image_name = input_batch[i]
        image_vector = output[i].flatten()

        near_db_engine.store_vector(image_vector, image_name)


engine = Engine(VECTOR_DIMENSION)

model = ResNet50(weights='imagenet', include_top=False)
image_list = get_image_list("sample_images/")
generate_vectors(model, image_list, engine)

test_image_list = get_image_list("test_images/")
generate_test_vectors(model, test_image_list, engine)