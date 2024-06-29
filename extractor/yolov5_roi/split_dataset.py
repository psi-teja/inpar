import os
import json, random, shutil, yaml
import pickle

current_dir = os.path.dirname(__file__)

# Define your dataset and folder paths
dataset_folder = os.path.join(current_dir, "datasets", "imerit")

images_folder = os.path.join(dataset_folder, "images")
labels_folder = os.path.join(dataset_folder, "labels")

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


# Path to the saved label encoder file
label_encoder_file = os.path.join(dataset_folder, "label_encoder.pkl")

# Check if the file exists and load the encoder, otherwise create a new one
if os.path.exists(label_encoder_file):
    with open(label_encoder_file, "rb") as f:
        label_encoder = pickle.load(f)


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

else:
    print("Error: label encoder not found. Failed to create training yml file")