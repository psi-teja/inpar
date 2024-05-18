import os
from tqdm import tqdm
import requests
from datetime import datetime
from pdf2image import convert_from_bytes

current_dir = os.path.dirname(__file__)


downloadFolder = os.path.join(current_dir, "datasets", "all")

imageFolder = os.path.join(downloadFolder, "images")
extractedDataFolder = os.path.join(downloadFolder, "extracted_data")


# Create directories if they don't exist
os.makedirs(imageFolder, exist_ok=True)
os.makedirs(extractedDataFolder, exist_ok=True)

subTableUrl = "https://o4xsvxdeo7p74yfscmfplihqqm0spsqn.lambda-url.ap-south-1.on.aws/get/SubTable"

# Send a GET request to the URL
response = requests.get(subTableUrl)


start_datetime = datetime.strptime('2024-04-17 14:24:39', '%Y-%m-%d %H:%M:%S')

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Extract data from the response
    data = response.json()  # Assuming the response is in JSON format
    # print(data)  # Print the data or process it further as needed

    for i in tqdm(range(len(data)), unit="sample"):
        
        sub = data[i]
        doc_id = sub.get('doc_id')

        sub_datetime = datetime.strptime(sub['inserted_time'], '%Y-%m-%d %H:%M:%S')

        if doc_id:
            fileUrl = f"https://abtzn7vmc5bae2auqjgtekhpiq0mehbs.lambda-url.ap-south-1.on.aws/get/document/{doc_id}"
            jsonUrl = f"https://abtzn7vmc5bae2auqjgtekhpiq0mehbs.lambda-url.ap-south-1.on.aws/get/ai_json/{doc_id}"
            file_response = requests.get(fileUrl)
            json_response = requests.get(jsonUrl)
            if file_response.status_code == 200 and json_response.status_code == 200:
                # Check if the response contains PDF data
                if 'application/pdf' in file_response.headers.get('content-type', ''):
                    pdf_bytes = file_response.content
                    # Convert the first page of the PDF to an image
                    first_page = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)[0]
                    # Save the image
                    image_path = f"{imageFolder}/{doc_id}.jpg"
                    first_page.save(image_path, "JPEG")
                    # print(f"First page of document {doc_id} saved as image: {image_path}")
                else:
                    # Save non-PDF files to the same directory
                    file_extension = file_response.headers['content-type'].split('/')[1]
                    file_path = f"{imageFolder}/{doc_id}.{file_extension}"
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    # print(f"File {doc_id}.{file_extension} downloaded successfully.")

                # Save JSON file
                json_path = f"{extractedDataFolder}/{doc_id}.json"
                with open(json_path, 'wb') as f:
                    f.write(json_response.content)
                # print(f"JSON file for document {doc_id} downloaded successfully.")
            else:
                pass
                # print(f"Error downloading {doc_id} details. File status code: {file_response.status_code}, JSON status code: {json_response.status_code}")
else:
    print("Error:", response.status_code)
