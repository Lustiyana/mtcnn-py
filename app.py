from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout, BatchNormalization
from keras_preprocessing import image as keras_img
import urllib.request
from urllib.request import urlopen
import ssl
from io import BytesIO
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def extract_face_from_image(img, required_size=(224, 224)):
    face_pixels = np.array(img)
    return extract_face(face_pixels, required_size)

def extract_face(image, required_size=(224, 224)):
    detector = MTCNN()
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


def are_images_equal(image_path1, image_path2):
    with urllib.request.urlopen(image_path2) as response:
        image_data = response.read()
    image1 = Image.open(image_path1)
    image2 = Image.open(BytesIO(image_data))

    if image1.size != image2.size:
        return False

    pixel_data1 = list(image1.getdata())
    pixel_data2 = list(image2.getdata())

    if pixel_data1 != pixel_data2:
        return False

    return True


@app.route('/process', methods=['POST'])
def process():
    try:
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
        model.load_weights('./model/model.h5')
        
        image = request.files['image']
        img = Image.open(image)

        face_pixels = extract_face_from_image(img)
        face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_GRAY2RGB)
        face_image = Image.fromarray(face_pixels_rgb)
        if 'image' not in request.files:
            return jsonify({'message': 'No image uploaded'}), 400

        x = keras_img.img_to_array(face_image)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=32)
        predicted_class = class_labels[np.argmax(pred)]

        specific_key = form_data.get('image_array')
        if len(specific_key) == 0:
            return jsonify({'success':True, 'message': ''})
        
        image_array= specific_key.split(';')

        result = []
        for image_path2 in image_array:
            result.append(are_images_equal(image, image_path2))

        if any(result):
            return jsonify({'success':False, 'message':'Gambar yang sudah diupload tidak boleh diupload kembali'}), 400
        else:
            return jsonify({'prediction': predicted_class, 'success':True, 'message':''})

    except Exception:
        return jsonify({'message': 'Absen gagal! Tidak ada wajah yang terdeteksi', 'success': False}), 400


if __name__ == '__main__':
    app.run(debug=True, port=8001)
