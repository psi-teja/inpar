from torch.utils.data import Dataset
import os, json, boto3
from PIL import Image
import numpy as np
import cv2
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer
import math, torch

processor_without_ocr = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
processor_with_ocr = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=True)
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def calculate_ioa(l1, t1, w1, h1, l2, t2, w2, h2):
    # Calculate coordinates of intersection rectangle
    x_left = max(l1, l2)
    y_top = max(t1, t2)
    x_right = min(l1 + w1, l2 + w2)
    y_bottom = min(t1 + h1, t2 + h2)

    # Check if there's no intersection (one or both rectangles have zero area)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of each rectangle
    area1 = w1 * h1
    area2 = w2 * h2

    if not area1 or not area2:
        return 0

    ioa = intersection_area / area1

    return ioa

def calculate_iou(l1, t1, w1, h1, l2, t2, w2, h2):
    # Calculate coordinates of intersection rectangle
    x_left = max(l1, l2)
    y_top = max(t1, t2)
    x_right = min(l1 + w1, l2 + w2)
    y_bottom = min(t1 + h1, t2 + h2)

    # Check if there's no intersection (one or both rectangles have zero area)
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of each rectangle
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate union area
    union_area = area1 + area2 - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area

    return iou


class CustomDataset_DP(Dataset):
    def __init__(self, image_folder, annotation_folder, label2id):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.processor = processor_without_ocr
        self.label2id = label2id

        self.sample_list = [file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_filename = self.sample_list[idx]
        doc_id = image_filename.split(".")[0]

        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path)

        annotation_path = os.path.join(self.annotation_folder, f"{doc_id}.json")
        with open(annotation_path, "r") as f:
            annotation_json = json.load(f)

        words = annotation_json["tokens"]
        boxes = annotation_json["bboxes"]

        # Convert floats to integers using nested list comprehension
        boxes = [[int(float_val) for float_val in sublist] for sublist in boxes]

        # Map over labels and convert to numeric id for each ner_tag
        ner_tags = [self.label2id[ner_tag] for ner_tag in annotation_json["ner_tags"]]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=ner_tags,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # return encoding

        model_input = {
            "input_ids": encoding.input_ids[0].to(device),
            "attention_mask": encoding.attention_mask[0].to(device),
            "bbox": encoding.bbox[0].to(device),
            "pixel_values": encoding.pixel_values[0].to(device),
        }

        return model_input, encoding.labels[0].to(device)

class CustomDataset_DDP(Dataset):
    def __init__(self, image_folder, annotation_folder, label2id):
        self.image_folder = image_folder
        self.annotation_folder = annotation_folder
        self.processor = processor_without_ocr
        self.label2id = label2id

        self.sample_list = [file for file in os.listdir(image_folder) if file.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_filename = self.sample_list[idx]
        doc_id = image_filename.split(".")[0]

        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path)

        annotation_path = os.path.join(self.annotation_folder, f"{doc_id}.json")
        with open(annotation_path, "r") as f:
            annotation_json = json.load(f)

        words = annotation_json["tokens"]
        boxes = annotation_json["bboxes"]

        # Convert floats to integers using nested list comprehension
        boxes = [[int(float_val) for float_val in sublist] for sublist in boxes]

        # Map over labels and convert to numeric id for each ner_tag
        ner_tags = [self.label2id[ner_tag] for ner_tag in annotation_json["ner_tags"]]

        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            word_labels=ner_tags,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # return encoding

        model_input = {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
            "bbox": encoding.bbox[0],
            "pixel_values": encoding.pixel_values[0],
        }

        return model_input, encoding.labels[0]


def preprocess_image_gray(image):

    image_gray = image.convert("L")
    return image_gray


# Preprocess the image
def preprocess_image_cv(image):

    # Convert to grayscale
    image_gray = image.convert("L")

    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image_gray), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    if image.mode != "RGB":
        image = image.convert("RGB")

    return blurred_image


def upload_folder_to_s3(local_folder_path, bucket_name, s3_folder_path):
    session = boto3.Session(profile_name='Developers_custom2_dev-pb-393436098818')

    # Create an S3 client using the session
    s3 = session.client('s3')
    # Walk through the local folder and upload each file to S3
    for root, dirs, files in os.walk(local_folder_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            # Construct the S3 key by removing the local folder path and joining with the S3 folder path
            s3_key = os.path.relpath(local_file_path, local_folder_path)
            s3_key = os.path.join(s3_folder_path, s3_key)

            s3_key = s3_key.replace("\\", "/")

            # Upload the file to S3
            s3.upload_file(local_file_path, bucket_name, s3_key)

    print(f"{colors.GREEN}Uploaded {local_folder_path} to s3://{bucket_name}/{s3_folder_path}{colors.END}")


def fraction_to_ratio(numerator, denominator):
    # Calculate the greatest common divisor (GCD)
    gcd = math.gcd(numerator, denominator)
    
    # Simplify the fraction
    simplified_numerator = numerator // gcd
    simplified_denominator = denominator // gcd
    
    # Return the simplified ratio as a string
    return f"{simplified_numerator}:{simplified_denominator}"


if __name__ == "__main__":
    iou = calculate_ioa(0, 0, 2, 2, 1, 1, 0.5, 0.5)

    local_folder_path = "AIBackend/DocAI/inpar-research/layoutLMv3/results/runs/20240422150953"

    # Specify the S3 bucket name
    bucket_name = "tally-ai-doc-ai-inpar-models"

    # Specify the S3 folder path
    s3_folder_path = "layoutLMv3_finetuned/20240420232435"

    upload_folder_to_s3(local_folder_path, bucket_name, s3_folder_path)

    print("IOU:", iou)