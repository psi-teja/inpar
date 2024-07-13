from flask import Flask, jsonify
import redis
import threading
import os
import cv2
import json
from io import BytesIO
from PIL import Image
from extractor_main import parse_invoice, merge_with_final_output
import logging
import requests
import fitz  # PyMuPDF
import numpy as np
import traceback
from pdf2image import convert_from_path
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = 'upload_doc'


# Define folder paths
save_data_url = "http://localhost:8000/db_connect/upload/json"  # Replace with your actual URL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def redis_listener():
    pubsub = redis_client.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = message['data'].decode('utf-8')
            request_data = json.loads(data)
            doc_id = request_data['doc_id']
            file_path = request_data['file_path']
            logger.info(f"Received doc_id: {doc_id}")

            tally_ai_json = {}

            try:
                # Determine file type
                file_ext = os.path.splitext(file_path)[1].lower()

                if file_ext == '.pdf':
                    # Process PDF, extract the first page
                    images = convert_from_path(file_path)

                    for i in range(len(images)):

                        pil_image = images[i]

                        # Convert PIL image to OpenCV image (numpy array)
                        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


                        tally_ai_json_page = parse_invoice(pil_image, cv2_image, pageNo = i+1)

                        
                        tally_ai_json = merge_with_final_output(tally_ai_json_page, tally_ai_json)

                else:
                    # Process image
                    pil_image = Image.open(file_path)
                    cv2_image = cv2.imread(file_path)

                    tally_ai_json = parse_invoice(pil_image, cv2_image, pageNo=1)

                if pil_image is None or cv2_image is None:
                    logger.error(f"Error opening file: {file_path}")
                    continue
                

                
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
                logger.error(f"Error processing file {doc_id}: {str(e)}")

@app.route('/')
def home():
    return jsonify({"message": "Redis subscriber server is running"})

if __name__ == '__main__':
    listener_thread = threading.Thread(target=redis_listener, daemon=True)
    listener_thread.start()
    app.run(port=5000)