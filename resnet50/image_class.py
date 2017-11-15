from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os.path


def get_image(image_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    im_path = os.path.join(script_dir, 'sample_images/' + image_name)
    return image.load_img(im_path, target_size=(224, 224))


def predict(img):
    model = ResNet50(weights='imagenet')

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return model.predict(x)


def process_image(image_name):
    img = get_image(image_name)
    preds = predict(img)

    # decode the results into a list of tuples (class, description, probability)
    print('Predicted:', decode_predictions(preds, top=3)[0])

process_image("cat.jpg")
process_image("dog.jpg")
process_image("plane.jpg")