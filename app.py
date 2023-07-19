from flask import Flask, render_template, request, jsonify
from PIL import Image
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import io
import base64
import numpy as np


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
    image = request.files['image']
    img = Image.open(image)
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('static/result.png')
    plt.close()
    face_pixels = extract_face_from_file("static/result.png")
    face_image = Image.fromarray(face_pixels)
    face_image.save("static/face_image.png")
    

    return render_template('result.html')

def get_data():
    data={
        'image': '/static/face_image.png'
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
