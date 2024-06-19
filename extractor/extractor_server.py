from flask import Flask, jsonify
import redis
import threading
import os
import cv2
import json
from PIL import Image
from extractor_main import parse_invoice
import logging
import requests
import numpy as np
import traceback
from pdf2image import convert_from_path


app = Flask(__name__)

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = 'upload_doc'

# Set environment variable for AWS profile
os.environ['AWS_PROFILE'] = "Developers_custom1_tallydev-AI-381491826341"

# Define folder paths
input_docs_folder = "AIBackend/DocAI/inpar-research/django_backend/input_docs"
output_jsons_folder = "AIBackend/DocAI/inpar-research/django_backend/output_jsons"
save_data_url = "http://localhost:8000/db_connect/upload/json"  # Replace with your actual URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def redis_listener():
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message['type'] == 'message':
            filename = message['data'].decode('utf-8')
            file_path = os.path.join(input_docs_folder, filename)
            logger.info(f"Received filename: {filename}")

            try:
                # Determine file type
                file_ext = os.path.splitext(filename)[1].lower()

                if file_ext == '.pdf':
                    # Process PDF, extract the first page
                    images = convert_from_path(file_path)

                    pil_image = images[0]

                    # Convert PIL image to OpenCV image (numpy array)
                    cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                else:
                    # Process image
                    pil_image = Image.open(file_path)
                    cv2_image = cv2.imread(file_path)

                if pil_image is None or cv2_image is None:
                    logger.error(f"Error opening file: {file_path}")
                    continue

                # Parse the invoice
                tally_ai_json = parse_invoice(pil_image, cv2_image)


                # Send JSON data to save_data endpoint
                doc_id = os.path.splitext(filename)[0]
                
                headers = {'Content-Type': 'application/json'}
                response = requests.post(
                    f"{save_data_url}/ai_json/{doc_id}/",
                    headers=headers,
                    data=json.dumps(tally_ai_json)
                )

                if response.status_code == 200:
                    logger.info(f"Successfully sent JSON to save_data endpoint for doc_id: {doc_id}")
                else:
                    logger.error(f"Failed to send JSON to save_data endpoint for doc_id: {doc_id}, Status Code: {response.status_code}, Response: {response.text}")

            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error processing file {filename}: {str(e)}")

@app.route('/')
def home():
    return jsonify({"message": "Redis subscriber server is running"})

if __name__ == '__main__':
    listener_thread = threading.Thread(target=redis_listener, daemon=True)
    listener_thread.start()
    app.run(port=5000)
