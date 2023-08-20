from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import tensorflow as tf
import requests
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout, BatchNormalization
from keras_preprocessing import image as keras_img

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def extract_face_from_file(filename, required_size=(160, 160)):
    image = Image.open(filename)
    return extract_face(image, required_size)

def extract_face(image, required_size=(160, 160)):
    detector = MTCNN()
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)

    x1, y1, width, height = results[0]['box']

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    face = pixels[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    gray_face = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)

    return gray_face

@app.route('/process', methods=['POST'])
def process():
    form_data = request.form
    specific_key = form_data.get('classes') 
    class_labels= specific_key.split(';')

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(len(class_labels),activation='softmax'))
    model.load_weights('./model/model_5.h5')
    
    image = request.files['image']
    img = Image.open(image)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('static/result.png')
    plt.close()
    face_pixels = extract_face_from_file("static/result.png")
    face_image = Image.fromarray(face_pixels)
    face_image.save("static/face_image.png")

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    img = keras_img.load_img("static/face_image.png", target_size=(224,224,3))
    x = keras_img.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    pred = model.predict(images, batch_size=32)
    predicted_class = class_labels[np.argmax(pred)]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=8001)
