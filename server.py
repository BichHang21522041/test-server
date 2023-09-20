from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from flask import render_template

# import numpy as np
# import cv2
# import base64
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import open_clip
from open_clip import tokenizer
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from collections import OrderedDict
from torchvision.datasets import CIFAR100
from torchvision import transforms





app = Flask(__name__)

model = torch.load("model_full.pth")
# model.to(device)
model.eval()
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# def preprocess():
#     model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
#     return preprocess

def _convert_to_rgb(image):
    return image.convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize(size=256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=(256, 256)),
    _convert_to_rgb,
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


# @app.route("/token", methods=['GET', 'POST'])
def token(text):
    tokenizer = open_clip.get_tokenizer('convnext_base_w')
    texts = tokenizer(text)
    return texts

# @app.route("/submit", methods = ['GET', 'POST'])
# def get_output():
# 	if request.method == 'POST':
# 		img = request.files['my_image']

# 		img_path = "static/" + img.filename	
# 		img.save(img_path)

# 		p = 'Chua biet'

# 	return render_template("index.html", prediction = p, img_path = img_path)


def building_feature(image):
    # Thêm một chiều cho batch_size
    image_input = image.unsqueeze(0)
    with torch.no_grad():
        image_feature = model.encode_image(image_input).float()
    return image_feature


def calculating_cosine(image_feature):
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    return image_feature

def handle_lable():
    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess, download=True)
    text_descriptions = [f"A photo of a {label}" for label in cifar100.classes]
    text_tokens = token(text_descriptions)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features, cifar100.classes



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename    
        img.save(img_path)

        # Open the image file
        img_pil = Image.open(img_path)
        
        # Preprocess the image
        img_tensor = preprocess(img_pil)

        img_feature = building_feature(img_tensor)

        img_feature_cal = calculating_cosine(img_feature)

        text_features, classes = handle_lable()

        text_probs = (100.0 * img_feature_cal @ text_features.T).softmax(dim=-1)

        top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

        p = [classes[index] for index in top_labels[0].numpy()]

    return render_template("index.html", prediction = p, img_path = img_path)

@app.route('/clip', methods=['POST', 'GET'] )

@cross_origin(origin='*')
def clip_process():
    return 'Hello'
