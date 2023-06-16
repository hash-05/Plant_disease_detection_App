from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os
from torchvision.transforms import ToTensor
from model import ResNet9
import pandas as pd

app = Flask(__name__)

model = ResNet9(in_channels=3, num_diseases=38)
model.load_state_dict(torch.load('plant-disease-model.pth', map_location=torch.device('cpu')))
model.eval()

df = pd.read_csv('Diseases.csv')


def preprocess_image(image):
    image = image.convert('RGB')  # Ensure image is in RGB format
    image = ToTensor()(image)  # Convert image to tensor
    image = image.unsqueeze(0)
    return image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print(filename)
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        print(file_path)
        image = Image.open(file)
        image = preprocess_image(image)
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_label = predicted[0].item()
        d = {'plant_name': df['Plant'][predicted_label],
             'disease_name': df['Disease'][predicted_label],
             'description': df['Description'][predicted_label],
             'solution': df['Solution'][predicted_label]}
        return render_template('index.html', data=d, img_path=file_path)


if __name__ == '__main__':
    app.run()
