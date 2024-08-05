# Description: This file contains the code to process images and extract features from them.


#libraries
import os
from PIL import Image
from flask import jsonify, request
import requests
import torch
import torchvision
from torchvision import transforms
import requests

# Load the pre-trained ResNet model

model = torchvision.models.resnet18(weights="DEFAULT")
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to extract features from an image
def extract_features(img):
    img = transform(img)
    with torch.no_grad():
        out = model(img.unsqueeze(0))
    output= out.squeeze().cpu().numpy().tolist()
    return output

# Function to process a list of image paths
def process_product_images(image_paths):
    try:
        print("The image paths:", image_paths)
        imgs = []
        
        # If image_paths is a string (single path), convert it to a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        for image_path in image_paths:
            print(f"Processing image path: {image_path}")
            try:
                if image_path.startswith('http'):
                    response = requests.get(image_path, stream=True, verify=False)
                    response.raise_for_status()
                    img = Image.open(response.raw).convert('RGB')
                else:
                    print(f"Checking local file: {image_path}")
                    if os.path.isfile(image_path):
                        print(f"Opening local file: {image_path}")
                        img = Image.open(image_path).convert('RGB')
                        print(f"Successfully opened local file: {image_path}")
                    else:
                        print(f"File not found: {image_path}")
                        continue
                imgs.append(img)
            except Exception as img_error:
                print(f"Error processing individual image {image_path}: {str(img_error)}")
        
        if not imgs:
            print("No valid images found")
            return None
        
        print(f"Processing {len(imgs)} images")
        features = [extract_features(img) for img in imgs]
        print(f"Extracted features for {len(features)} images")
        return features if features else None
    except Exception as e:
        print("Error in process_product_images:", str(e))
        return None
