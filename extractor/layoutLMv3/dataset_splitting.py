import os
import configparser
import random
import shutil

current_dir = os.path.dirname(__file__)

def clear_folder(folder):
    # Clear the contents of the folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def split_data(dataset_folder, train_ratio):

    print(f"Dataset Folder: {dataset_folder}")
    dataset_details_file = os.path.join(dataset_folder, "details.cfg")
    dataset_config = configparser.ConfigParser()
    dataset_config.optionxform = str
    dataset_config.read(dataset_details_file)
    num_samples = dataset_config["General"]["NumberOfSamples"]
    print(f"NumberOfSamples: {num_samples}")

    image_folder = os.path.join(dataset_folder, "images")
    label_folder = os.path.join(dataset_folder, "label_json")

    # Create train and test folders
    train_image_folder = os.path.join(dataset_folder, "train", "images")
    test_image_folder = os.path.join(dataset_folder, "test", "images")
    os.makedirs(train_image_folder, exist_ok=True)
    os.makedirs(test_image_folder, exist_ok=True)

    train_label_folder = os.path.join(dataset_folder, "train", "label_json")
    test_label_folder = os.path.join(dataset_folder, "test", "label_json")
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(test_label_folder, exist_ok=True)

    # Clear train and test folders
    clear_folder(train_image_folder)
    clear_folder(test_image_folder)
    clear_folder(train_label_folder)
    clear_folder(test_label_folder)

    # Get list of image and label files
    image_files = os.listdir(image_folder)
    label_files = os.listdir(label_folder)

    # Randomly shuffle the files
    random.shuffle(image_files)

    # Split the files into train and test sets
    num_train = int(len(image_files) * train_ratio)
    train_images = image_files[:num_train]
    test_images = image_files[num_train:]

    dataset_config["General"]["TrainSamples"] = str(len(train_images))
    dataset_config["General"]["TestSamples"] = str(len(test_images))

    # Move train images and labels to train folders
    for image in train_images:
        shutil.copy(os.path.join(image_folder, image), os.path.join(train_image_folder, image))
        label = image.replace(".jpg", ".json")  # Assuming labels have the same name as images with a different extension
        shutil.copy(os.path.join(label_folder, label), os.path.join(train_label_folder, label))

    # Move test images and labels to test folders
    for image in test_images:
        shutil.copy(os.path.join(image_folder, image), os.path.join(test_image_folder, image))
        label = image.replace(".jpg", ".json")  # Assuming labels have the same name as images with a different extension
        shutil.copy(os.path.join(label_folder, label), os.path.join(test_label_folder, label))

    with open(os.path.join(dataset_folder, "details.cfg"), "w") as configfile:
        dataset_config.write(configfile)


if __name__ == "__main__":

    dataset_folder = os.path.join(current_dir, "datasets", "all")

    split_data(dataset_folder, train_ratio=0.6)
