import os
import json
from shutil import copyfile
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import json
from tqdm import tqdm
from layoutLMv3_utils import convert_labels, ledger_fields_list

current_dir = os.path.dirname(__file__)

# Define your dataset and folder paths
dataset_folder = os.path.join(current_dir, "datasets", "imerit")

phase = "phase1"

phase_folder = os.path.join(dataset_folder, phase)

doc_folder = os.path.join(phase_folder, "docs")
individualfield_json_folder = os.path.join(phase_folder, "individualfield_jsons")

images_folder = os.path.join(phase_folder, "images")
labels_folder = os.path.join(phase_folder, "tally_ai_jsons")

# Create directories if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

# Variables to store the number of samples and label counts
num_of_samples = 0

missing_jsons = []

doc_list = os.listdir(doc_folder)

# Loop through documents in doc_folder
for i in tqdm(range(len(doc_list)), unit="sample"):
    doc_name = doc_list[i]
    doc_path = os.path.join(doc_folder, doc_name)
    individualfield_json_path = os.path.join(individualfield_json_folder, doc_name.split(".")[0] + "_IndividualField.json")
    
    with open(individualfield_json_path, 'r') as f:
        individualfield_data = json.load(f)

    # Check if the document is a PDF or an image
    if doc_name.endswith(".pdf"):
        # PDF processing logic
        with open(doc_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            # Loop through each page of the PDF
            for page_num in range(len(pdf_reader.pages)):

                json_path = os.path.join(labels_folder, f"{doc_name.split('.')[0]}_page{page_num+1}.json")

                if os.path.exists(json_path):
                    num_of_samples += 1
                    continue
        
                try:
                    individualfield_data_page = individualfield_data[page_num]
                    num_of_samples += 1
                except:
                    missing_jsons.append(f"{doc_name.split('.')[0]}_page{page_num+1}")
                    continue

                individualfield_data_page = convert_labels(individualfield_data_page)


                with open(json_path, "w") as f:
                    json.dump(individualfield_data_page, f)

                    # Extract each page as an image
                page_image = convert_from_path(doc_path, first_page=page_num+1, last_page=page_num+1)[0]
                # Save the page image to images_folder
                page_image.save(os.path.join(images_folder, f"{doc_name.split('.')[0]}_page{page_num+1}.jpeg"))

                
        
    elif doc_name.endswith((".jpeg", ".png", ".jpg")):
        # Image processing logic
        # Copy the image to images_folder
        copyfile(doc_path, os.path.join(images_folder, doc_name))

        json_path = os.path.join(labels_folder, f"{doc_name.split('.')[0]}.json")

        if os.path.exists(json_path):
            num_of_samples += 1
            continue
        
        individualfield_data_page = individualfield_data[0]
        individualfield_data_page = convert_labels(individualfield_data_page)
        with open(json_path, "w") as f:
            json.dump(individualfield_data_page, f)
        num_of_samples += 1

    else:
        continue


print(f"Number of Samples: {num_of_samples}")
print(f"Missing Jsons:")
for file in missing_jsons:
    print(file)