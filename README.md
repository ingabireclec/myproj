# Visual Search API

This project implements a visual search API using Flask, allowing users to search for similar products based on image similarity. It uses a pre-trained ResNet model for feature extraction and cosine similarity for matching.

## Features

- Fetch and process product images from an external API
- Extract feature vectors from product images
- Save feature vectors to a database
- Search for similar products using image uploads
- RESTful API endpoints for various operations

## Prerequisites

- Python 3.10+
- pip (Python package installer)
- Flask
- PyTorch
- torchvision
- Pillow
- NumPy
- Requests
- tensorflow

## Installation

1. Install Python:
   - Download and install Python 3.7 or later from [python.org](https://www.python.org/downloads/)
   - Ensure that Python is added to your system PATH

2. Install pip (if not installed with Python):
   - Download [get-pip.py](https://bootstrap.pypa.io/get-pip.py)
   - Run `python get-pip.py`

3. Clone the repository:
   ```
   git clone https://github.com/PUNDAGROUP/izitini_ai_ml.git
   cd iziti_ai_ml
   ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Install PyTorch:
   - Visit [PyTorch's official website](https://pytorch.org/get-started/locally/)
   - Select your preferences (OS, package manager, Python, CUDA version)
   - Run the provided command, for example:
     ```
     pip3 install torch torchvision torchaudio
     ```

$ mkvirtualenv myvirtualenv --python=/usr/bin/python3.11


## Usage

1. Start the Flask server:
   ```
   python app.py
   ```

2. The server will start running on `http://localhost:5000`

### API Endpoints

- `GET /fetch_and_process_products`: Fetch products from the external API and process their images
- `POST /add_new_product`: Add a new product's feature vector to the database
- `POST /search`: Search for similar products by uploading an image
- `GET /get-all_feature_vectors`: Retrieve all feature vectors from the database

### Testing the Search Functionality

To test the visual search:

1. Use the `/search` endpoint with a POST request, including an image file.
2. The API will return a list of similar product IDs.
3. You can then use these IDs to fetch the corresponding product details and image URLs from the external API.

## Project Structure

- `app.py`: Main Flask application
- `image_processing.py`: Image processing and feature extraction functions
- `database.py`: Database operations and API interactions
