import os
import json
from shutil import copyfile
from tqdm import tqdm
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import json, random, shutil, yaml
from yolov5_roi_utils import YOLOv5LabelEncoder  # Import the YOLOv5LabelEncoder class

current_dir = os.path.dirname(__file__)

# Define your dataset and folder paths
dataset_folder = os.path.join(current_dir, "datasets", "imerit")

doc_folder = os.path.join(dataset_folder, "docs")
roi_json_folder = os.path.join(dataset_folder, "roi_jsons")

images_folder = os.path.join(dataset_folder, "images")
labels_folder = os.path.join(dataset_folder, "labels")

# Create directories if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

label_encoder = YOLOv5LabelEncoder()

# Variables to store the number of samples and label counts
num_of_samples = 0
missing_jsons = []
label_count = {}

doc_list = os.listdir(doc_folder)

# Loop through documents in doc_folder
for i in tqdm(range(len(doc_list)), unit="sample"):
    doc_name = doc_list[i]
    doc_path = os.path.join(doc_folder, doc_name)
    roi_json_path = os.path.join(roi_json_folder, doc_name.split(".")[0] + "_ROI.json")
    
    with open(roi_json_path, 'r') as f:
        roi_data = json.load(f)

    # Check if the document is a PDF or an image
    if doc_name.endswith(".pdf"):
        # PDF processing logic
        with open(doc_path, 'rb') as pdf_file:
            pdf_reader = PdfReader(pdf_file)
            # Loop through each page of the PDF
            for page_num in range(len(pdf_reader.pages)):

                label_file = open(os.path.join(labels_folder, f"{doc_name.split('.')[0]}_page{page_num+1}.txt"), "w")

                # Extract each page as an image
                page_image = convert_from_path(doc_path, first_page=page_num+1, last_page=page_num+1)[0]
                # Save the page image to images_folder
                page_image.save(os.path.join(images_folder, f"{doc_name.split('.')[0]}_page{page_num+1}.jpg"))
        
                
                try:
                    roi_data_page = roi_data[page_num]["ROI"]
                    num_of_samples += 1
                except:
                    missing_jsons.append(f"{doc_name.split('.')[0]}_page{page_num+1}")
                    label_file.close()
                    continue

                for label_dict in roi_data_page:
                    
                    for label in label_dict:
                        # Update label counts
                        if label not in label_count:
                            label_count[label] = 1
                        else:
                            label_count[label] += 1
                        
                        label_encoder.add_label(label)
                        l, t, w, h = label_dict[label]['location']['ltwh']
                        x,y = l + w/2, t + h/2
                        label_file.write(f"{label_encoder.get_label_id(label)} {x} {y} {w} {h}\n")
                
                label_file.close()
        
    elif doc_name.endswith((".jpeg", ".png", ".jpg")):
        # Image processing logic
        # Copy the image to images_folder
        copyfile(doc_path, os.path.join(images_folder, doc_name))

        label_file = open(os.path.join(labels_folder, f"{doc_name.split('.')[0]}.txt"), "w")

        roi_data_page = roi_data[0]["ROI"]

        for label_dict in roi_data_page:
            
            for label in label_dict:
                # Update label counts
                if label not in label_count:
                    label_count[label] = 1
                else:
                    label_count[label] += 1
                
                label_encoder.add_label(label)
                l, t, w, h = label_dict[label]['location']['ltwh']
                x,y = l + w/2, t + h/2
                label_file.write(f"{label_encoder.get_label_id(label)} {x} {y} {w} {h}\n")
        num_of_samples += 1
        label_file.close()

        num_of_samples +=1
    else:
        continue

# Write details to a file
details_file = os.path.join(dataset_folder, "details.cfg")
with open(details_file, "w") as f:
    f.write("[General]\n")
    f.write(f"NumberOfSamples = {num_of_samples}\n\n")
    f.write("[LabelCounts]\n")
    for label in label_count:
        f.write(f"{label} = {label_count[label]}\n")



train_image_dir = os.path.join(images_folder, "train")
val_image_dir = os.path.join(images_folder, "val")
test_image_dir = os.path.join(images_folder, "test")

train_label_dir = os.path.join(labels_folder, "train")
val_label_dir = os.path.join(labels_folder, "val")
test_label_dir = os.path.join(labels_folder, "test")

for directory in [train_image_dir, val_image_dir, test_image_dir, train_label_dir, val_label_dir, test_label_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)

# Get list of image files
image_files = [file for file in os.listdir(images_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Split the dataset into train, val, and test
random.shuffle(image_files)
train_split = int(0.7 * len(image_files))
val_split = int(0.15 * len(image_files))

train_images = image_files[:train_split]
val_images = image_files[train_split:train_split+val_split]
test_images = image_files[train_split+val_split:]

# Move images and labels to respective directories
for image in train_images:
    shutil.move(os.path.join(images_folder, image), os.path.join(train_image_dir, image))
    shutil.move(os.path.join(labels_folder, image.split(".")[0]+".txt"), os.path.join(train_label_dir, image.split(".")[0]+".txt"))

for image in val_images:
    shutil.move(os.path.join(images_folder, image), os.path.join(val_image_dir, image))
    shutil.move(os.path.join(labels_folder, image.split(".")[0]+".txt"), os.path.join(val_label_dir, image.split(".")[0]+".txt"))

for image in test_images:
    shutil.move(os.path.join(images_folder, image), os.path.join(test_image_dir, image))
    shutil.move(os.path.join(labels_folder, image.split(".")[0]+".txt"), os.path.join(test_label_dir, image.split(".")[0]+".txt"))



# Write to YAML file
yml_data = {
    "path": "../datasets/imerit",  # dataset root dir
    "train": "images/train",  # train images (relative to 'path') 128 images
    "val": "images/val",  # val images (relative to 'path') 128 images
    "test": "images/test",  # test images (optional)
    "names": label_encoder.get_id_dict()  # Classes
}

yml_file_path = os.path.join(dataset_folder, "roi.yml")
with open(yml_file_path, 'w') as label_file:
    yaml.dump(yml_data, label_file)