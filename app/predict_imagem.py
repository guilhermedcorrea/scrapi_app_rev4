
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import nltk
import spacy

model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.save("model.keras")
loaded_model = tf.keras.saving.load_model("model.keras")
x = tf.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))



def detect_objects(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (416, 416))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)


    bounding_boxes_and_classes = []
    for prediction in predictions[0]:
        x, y, w, h = prediction[:4]
        confidence = prediction[4]
        class_id = np.argmax(prediction[5:]) + 5  
        if confidence > 0.5:  
            bounding_boxes_and_classes.append((x, y, w, h, class_id))
    
    return bounding_boxes_and_classes

image_path = ''


results = detect_objects(image_path)


for box, class_id in results:
    print(f"Classe: {class_id}, Bounding Box: {box}")



image_path = ''


results = detect_objects(image_path)


for box, class_id in results:
    print(f"Classe: {class_id}, Bounding Box: {box}")