# database.py
import os
import traceback
from urllib.parse import urljoin
from flask import json
import numpy as np
from torch import cosine_similarity
import torch
import requests
from image_processing import process_product_images
import urllib3

#disable ssl warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
feature_vec_list = []

'''  the function to fetch and process products from the API endpoint.'''

def fetch_and_process_products(limit=10):
    base_url = "https://dev.izitini.com/"
    api_url = urljoin(base_url, "api/products/")
    view_url = "https://dev.izitini.com/storage/"
    try:
        response = requests.get(api_url, verify=False)  
        response.raise_for_status()  
        products = response.json()[:limit]
        
        processed_count = 0
        failed_count = 0
        failed_images = []

        total_products = len(products)
        #print(f"Total products to process: {total_products}")

        for product in products:
            product_id = product.get('id')
            image_paths = product.get('images', []) + [product.get('image')]
            image_paths = [urljoin(view_url, image_path) for image_path in image_paths if image_path]
            # print(f"Processing product {product_id}, image paths: {image_paths}")
            
            features = process_product_images(image_paths)
            if features is not None:

                # Save the feature vector to the database
                save_feature_vector(product_id, features)
                processed_count += 1
                # print(f"Successfully processed product {product_id}")
            else:
                failed_count += 1
                failed_images.append(image_paths)
                # print(f"Failed to process product {product_id}")
            
            # print(f"Progress: {processed_count + failed_count}/{total_products}")

        return {
            "status": "success", 
            "total_products": total_products,
            "processed_count": processed_count, 
            "failed_count": failed_count, 
            "failed_images": failed_images
        }
    except requests.RequestException as e:
        print("Error:", e)
        return {"status": "error", "message": f"Failed to fetch products: {str(e)}"} 

'''Define the function to save the feature vector to the database'''
def save_feature_vector(product_id, features):
    feature_vec_list.append((product_id, features))
    api_endpoint = 'https://izitini.com/api/products'  # Remove the query parameter from the URL
    save_feature_as_file(features, api_callback, api_endpoint, product_id)
    
    return {"product_id": product_id, "status": "success"}



def get_all_feature_vectors():
    """
    Fetch all feature vectors from the API endpoint.
    
    Returns:
    list: A list of tuples containing (product_id, features) for each product.
    """
    api_endpoint = 'https://izitini.com/api/products/vectors'
    print("Starting get_all_feature_vectors function")

    
    try:
        # print(f"Fetching data from {api_endpoint}")
        response = requests.get(api_endpoint)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        # print(f"Received {len(data)} items from API")
        
        feature_vectors = []
        for item in data:
            product_id = item['product_id']
            features = item['feature_vector']
            # Convert features from string to list if necessary
            if isinstance(features, str):
                features = json.loads(features)
            
            # Convert features to numpy array for easier processing
            features = np.array(features)
            
            # print(f"Feature vector shape for product {product_id}: {features.shape}")
            
            feature_vectors.append((product_id, features))
        
        # print(f"Finished processing {len(feature_vectors)} feature vectors")
        return feature_vectors
    
    except requests.RequestException as e:
        print(f"Error fetching feature vectors: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return []
    

'''Define the function to search for similar products based on a query feature vector'''


def search_feature_vector(query_features, threshold=0.5, top_k=10):   
    results = []
    # print("Query features type:", type(query_features))
    # print("Query features shape:", np.array(query_features).shape)
    
    # Ensure query_features is a 2D tensor
    query_features_tensor = torch.tensor(query_features, dtype=torch.float32)
    if query_features_tensor.dim() == 1:
        query_features_tensor = query_features_tensor.unsqueeze(0)
    
    feature_vec_list = get_all_feature_vectors()
    # print(f"Retrieved {len(feature_vec_list)} feature vectors for comparison")

    for product in feature_vec_list:
        product_id = product[0]
        product_features = product[1]
        
        # Ensure product_features is a 2D tensor
        product_features_tensor = torch.tensor(product_features, dtype=torch.float32)
        if product_features_tensor.dim() == 1:
            product_features_tensor = product_features_tensor.unsqueeze(0)
        
        # print(f"Product {product_id} features shape: {product_features_tensor.shape}")
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(query_features_tensor.unsqueeze(1), 
                                         product_features_tensor.unsqueeze(0), dim=2)
        
        max_similarity = similarity.max().item()
        print(f"Max similarity for product {product_id}: {max_similarity}")
        
        if max_similarity >= threshold:
            results.append((product_id, max_similarity))
        
        if len(results) >= top_k:
            break
    
    # Sort results by similarity (highest first) and take top_k
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:top_k]
    
    print(f"Top {len(top_results)} results:")
    for product_id, sim in top_results:
        print(f"Product {product_id}: Similarity {sim}")
    
    return [product_id for product_id, _ in top_results]

def get_product_details(product_id):
    product = next((x for x in feature_vec_list if x[0] == product_id), None)
    return {"product_id": product[0], "features": product[1]} if product else None



def save_feature_as_file(features, callback, api_endpoint, product_id):
    """
    Save the feature data to a file and execute the callback function.
    
    Parameters:
    features (any): The feature data to save.
    callback (function): The callback function to execute after saving the file.
    api_endpoint (str): The API endpoint to send the data to.
    product_id (str): The product ID to associate with the feature data.
    """
    # print('printing features:', features)
    # Convert features to a string format
    vector_txt = str(features)

    # print(f"Feature vector for product {product_id}:\n{vector_txt}")
    # Define the output file path
    output_file = f'vector_{product_id}.txt'

    # Write the data to a file
    try:
        with open(output_file, 'w') as file:
            file.write(vector_txt)
        print(f"File '{output_file}' created successfully.")
    except Exception as e:
        print(f"Error creating file '{output_file}': {str(e)}")
        return {"status": "error", "message": f"Error creating file '{output_file}': {str(e)}"}

    # Execute the callback function with the output file path and additional arguments
    callback(output_file, api_endpoint, product_id)

def api_callback(file_path, api_endpoint, product_id):
    """
    Send feature data file to the API endpoint along with the product ID.

    Parameters:
    file_path (str): The path to the file containing the feature data.
    api_endpoint (str): The API endpoint to send the data to.
    product_id (str): The product ID to associate with the feature data.
    """
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"File '{file_path}' does not exist"}

        # Prepare the multipart form data
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file, 'text/plain')}
            data = {'id': product_id}

            headers = {
                'Authorization': 'Bearer YOUR_API_TOKEN'  # Update with your actual token
            }

            # Construct the full URL
            full_url = f"{api_endpoint}/{product_id}"

            # print(f"Sending POST request to URL: {full_url}")
            # print(f"Uploading file: {file_path}")
            # print(f"Product ID: {product_id}")

            response = requests.post(full_url, files=files, data=data, headers=headers)
            response.raise_for_status()
            
            print('Response:', response.text)
            print(f"Data sent successfully for product {product_id}")

        # Delete the file after successful API call
        os.remove(file_path)
        # print(f"File '{file_path}' deleted successfully.")

        return {"status": "success", "message": f"Feature vector file for product {product_id} uploaded to API"}

    except requests.RequestException as e:
        error_message = f"Error sending data for product {product_id}: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f" Response: {e.response.text}"
        print(error_message)
        return {"status": "error", "message": error_message}
    except Exception as e:
        error_message = f"Unexpected error for product {product_id}: {str(e)}"
        print(error_message)
        return {"status": "error", "message": error_message}
#
# Function to delete a file after it has been sent to the API


def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} has been deleted successfully.")
    except Exception as e:
        print(f"Error deleting file {file_path}: {str(e)}")