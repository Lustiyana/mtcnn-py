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
import urllib.request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


def extract_face_from_image(img, required_size=(224, 224)):
    face_pixels = np.array(img)
    return extract_face(face_pixels, required_size)

def extract_face(image, required_size=(224, 224)):
    detector = MTCNN()
    # image = image.convert('RGB')
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
    # class_labels = ['Adi Pratama Putra', 'Afrinaldi', 'Aka Fadila', 'Alia Fadhila', 'Ammar Taradifa', 'Andi Malia Fadilah Bahari', 'Arief Raihan Nur Rahman', 'Arif Muhamad Iqbal ', 'Ashrul Nasrulloh', 'Azhar', 'Aziz Nuzul Praramadhana', 'Bella Arsylia ', 'Elis Kartika', 'Faiq Fadhlurrahman El Hakim', 'Farhan Rizky', 'Fawzan Ibnu Fajar', 'Fellia Ayu S A', 'Gani', 'Hilman Suhendar', 'Ifany Dewi Tustianti', 'Ilham Rizky Agustin ', 'Imam Firdaus', 'Indah Sri Lestari', 'Indri Nurfiani', 'Intan Permata Sari', 'Iqbal Putra Ramadhan', 'Jalalul', 'Laela Chintia Alviani ', 'Lustiyana', 'Moch Apip Tanuwijaya', 'Moch Arsyil Albany', 'Mochamad Najib Budi Noosrsyahbannie', 'Muhamad Farid Fauzi', 'Muhamad Iqbal Setiawan', 'Muhamad Rizki Isa Darmawan', 'Muhammad Afian Anwar', 'Muhammad Alwy Solehudin', 'Muhammad Fahmi Rizaldi Ilham', 'Muhammad Gilang Nur Haliz', 'Mujahid Ansori Majid', 'Naufal Rizqullah', 'Nurdila Farha Kamila', 'Nurul Aulia Dewi', 'Raihan Adam', 'Rifaldo Sukma Hidayat', 'Siti Haerani', 'Siti Yayah Rokayah ', 'Soniawan', 'fahriz']


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
    # face_pixels = extract_face_from_image(img)
    # face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_GRAY2RGB)
    # face_image = Image.fromarray(face_pixels_rgb)
    try:
        face_pixels = extract_face_from_image(img)
        face_pixels_rgb = cv2.cvtColor(face_pixels, cv2.COLOR_GRAY2RGB)
        face_image = Image.fromarray(face_pixels_rgb)
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        x = keras_img.img_to_array(face_image)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        pred = model.predict(images, batch_size=32)
        predicted_class = class_labels[np.argmax(pred)]
        return jsonify({'prediction': predicted_class, 'success':True})
    except Exception:
        return jsonify({'error': 'Absen gagal! Tidak ada wajah yang terdeteksi, coba lagi', 'success': False})


def are_images_equal(image_path1, image_path2):
    urllib.request.urlretrieve(image_path2, 'image.jpeg')
    image1 = Image.open(image_path1)
    image2 = Image.open('image.jpeg')

    if image1.size != image2.size:
        return False

    pixel_data1 = list(image1.getdata())
    pixel_data2 = list(image2.getdata())

    if pixel_data1 != pixel_data2:
        return False

    return True

@app.route('/image-checking', methods=['POST'])
def image_checking():
    form_data = request.form
    image_path1 = request.files['image']
    specific_key = form_data.get('image_array')
    if len(specific_key) == 0:
        return jsonify({'success':True, 'message': ''})
        
    image_array= specific_key.split(';')

    result = []
    for image_path2 in image_array:
        if are_images_equal(image_path1, image_path2):
            result.append(True)
        else:
            result.append(False)


    if any(result):
        return jsonify({'success':False, 'message':'Gambar yang sudah diupload tidak boleh diupload kembali'})
    else:
        return jsonify({'success':True, 'message': ''})

if __name__ == '__main__':
    app.run(debug=True, port=8001)
