from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os.path
from os import listdir


BATCH_SIZE = 32


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




model = ResNet50(weights='imagenet')
image_list = get_image_list("sample_images/")

# counting only complete batches for now
for batch in range(len(image_list) / BATCH_SIZE):
    input_batch = []
    for i in range(BATCH_SIZE):
        input_batch.append(get_image(image_list[BATCH_SIZE * batch + i]))

    input_array = np.array(input_batch)

    preds = predict(model, input_array)

    print('Predicted:', decode_predictions(preds, top=3))

