
#app.py
import os
from flask import Flask, request, jsonify, render_template
from image_processing import process_product_images
from database import save_feature_vector, get_all_feature_vectors,fetch_and_process_products, search_feature_vector

from PIL import Image
import numpy as np
app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')


'''a function  to fetch and process products from the database 
and then save the feature vectors in the database'''
@app.route('/fetch_and_process_products', methods=['GET'])
def fetch_and_process():
    result = fetch_and_process_products()
    return jsonify(result)

    

 #a route to add new product features to the database   
@app.route('/add_new_product', methods=['POST'])
def add_new_product_features():
    ''' parameters: product_id: str, image_urls: list of str

    '''
    product_image_urls = request.form.get("image_urls")
    product_id = request.form.get('product_id')

    if not product_id:
        return jsonify({"error": "No product ID provided"}), 400
    
    if not product_image_urls:
        return jsonify({"error": "No image URL provided"}), 400
    
    features = process_product_images(product_image_urls)
    
    result=save_feature_vector(product_id, features)

    return jsonify(result)

'''a function to search for similar products in the database'''
@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    temp_path = save_file(file, 'temp_search')
    
    query_features = process_product_images(temp_path)
    if isinstance(query_features, list):
        query_features = np.array(query_features)
    
    top_results = search_feature_vector(query_features, threshold=0.5)
    return jsonify({"results": top_results})

# Function to save the uploaded file
def save_file(file, filename):
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, f'{filename}.jpg')
    file.save(file_path)
    return file_path

# Function to get all feature vectors from the database
@app.route('/get-all_feature_vectors', methods=['GET'])

def get_all_feature_vectors():
    all_vectors = get_all_feature_vectors()
    return jsonify({"feature_vectors": all_vectors})

if __name__ == '__main__':
    app.run(debug=True)
