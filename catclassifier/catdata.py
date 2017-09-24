from PIL import Image
import numpy as np
from os import listdir
import random

IMAGE_SIZE = 64*64*3

def load_image(filename):
    # print (filename)
    img = Image.open(filename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return np.reshape(data, (IMAGE_SIZE))


def get_cat_file_names():
    cat_path = "cats/"
    return [cat_path + f for f in listdir(cat_path) if f.endswith(".jpg")]


def get_noncat_file_names():
    noncats_path = "noncats/"
    return [noncats_path + f for f in listdir(noncats_path) if f.endswith(".jpg")]


def extend_file_name_list(cat_files, noncat_files, size):
    train_files = []
    while len(train_files) < size:
        train_files.extend(cat_files)
        train_files.extend(noncat_files)
    return train_files


def generate_minibatch_paths(size):
    train_files = extend_file_name_list(
        get_cat_file_names(),
        get_noncat_file_names(),
        size
    )
    return random.sample(train_files, size)


def generate_minibatch(size):
    X = []
    Y = []
    random.seed()
    train_files = generate_minibatch_paths(size)

    for file_name in train_files:
        x = load_image(file_name)
        if (file_name.startswith("cats/")):
            y = [1, 0]
        else:
            y = [0, 1]

        X.append(x)
        Y.append(y)

    return train_files, \
           np.reshape(np.asarray(X), (IMAGE_SIZE, size)), \
           np.reshape(np.asarray(Y), (2, size))
