import re
from torch.utils.data import Dataset
import os, json, boto3
from PIL import Image, ImageDraw
import numpy as np
import cv2, io
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer
import math, torch
import torch.nn.functional as F
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import zipfile
import easyocr


processor_without_ocr = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=False
)
processor_with_ocr = LayoutLMv3Processor.from_pretrained(
    "microsoft/layoutlmv3-base", apply_ocr=True
)
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


current_dir = os.path.dirname(__file__)

class ProcessorWithEASYOCR:
    def __init__(self) -> None:
        if device == "cuda:0":
            self.easy_ocr_reader = easyocr.Reader(['en'], gpu=True)
        else:
            self.easy_ocr_reader = easyocr.Reader(['en'])

    
    def get_easy_ocr_data(self, image_bytes, image_width, image_height):
        # Read the image bytes with EasyOCR
        results = self.easy_ocr_reader.readtext(image_bytes, detail=1, link_threshold=0.1)

        texts = []
        bboxes = []
        for result in results:
            bbox, text, confidence = result
            texts.append(text)
            # Convert bbox from [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] to [left, top, width, height]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            left = min(x_coords)
            top = min(y_coords)
            width = max(x_coords) - left
            height = max(y_coords) - top
            # Normalize to 0-1000 range
            left = (left / image_width)
            top = (top / image_height)
            width = (width / image_width)
            height = (height / image_height)
            bboxes.append([left, top, width, height])
        
        return {"texts": texts, "bboxes_ltwh": bboxes}
    

    def get_encodings(self, pil_image):
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        image_width, image_height = pil_image.size

        # Get OCR data from AWS Textract
        ocr_data = self.get_easy_ocr_data(image_bytes, image_width, image_height)

        scaled_bboxes_xyxy = []

        pil_image_with_bboxes = pil_image.copy()
        draw = ImageDraw.Draw(pil_image_with_bboxes)

        for bbox in ocr_data["bboxes_ltwh"]:
            l, t, w, h = bbox

            x1 = l*image_width
            y1 = t*image_height
            x2 = (l+w)*image_width
            y2 = (t+h)*image_height

            draw.rectangle([x1,y1,x2,y2], outline="red", width=3)

            scaled_x1 = int(l * 1000)
            scaled_y1 = int(t * 1000)
            scaled_x2 = int((l + w) * 1000)
            scaled_y2 = int((t + h) * 1000)

            scaled_bboxes_xyxy.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

        output_path = os.path.join(current_dir, "ocr_bboxes.jpg")
        pil_image_with_bboxes.save(output_path)

        encodings = processor_without_ocr(
            pil_image,
            text=ocr_data["texts"],
            boxes=scaled_bboxes_xyxy,
            return_tensors="pt",
        )
        return encodings

class ProcessorWithAWSOCR:
    def __init__(self, selected_level) -> None:
        self.textract = boto3.client("textract", region_name="ap-south-1")
        self.level = selected_level

    def get_aws_ocr_data(self, imageBytes):
        response = self.textract.detect_document_text(Document={"Bytes": imageBytes})
        texts = []
        bboxes = []
        for item in response["Blocks"]:
            if item["BlockType"] == self.level:
                texts.append(item["Text"])
                bboxes.append(
                    [
                        item["Geometry"]["BoundingBox"]["Left"],
                        item["Geometry"]["BoundingBox"]["Top"],
                        item["Geometry"]["BoundingBox"]["Width"],
                        item["Geometry"]["BoundingBox"]["Height"],
                    ]
                )
        return {"texts": texts, "bboxes_ltwh": bboxes}

    def get_encodings(self, pil_image):
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        image_bytes = img_byte_arr.getvalue()

        # Get OCR data from AWS Textract
        ocr_data = self.get_aws_ocr_data(image_bytes)

        scaled_bboxes_xyxy = []

        for bbox in ocr_data["bboxes_ltwh"]:
            l, t, w, h = bbox

            scaled_x1 = int(l * 1000)
            scaled_y1 = int(t * 1000)
            scaled_x2 = int((l + w) * 1000)
            scaled_y2 = int((t + h) * 1000)

            scaled_bboxes_xyxy.append([scaled_x1, scaled_y1, scaled_x2, scaled_y2])

        # Get encodings using the processor without built-in OCR
        encodings = processor_without_ocr(
            pil_image,
            text=ocr_data["texts"],
            boxes=scaled_bboxes_xyxy,
            return_tensors="pt",
        )
        return encodings


class colors:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


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

    if area1 < area2:
        ioa = intersection_area / area1
    else:
        ioa = intersection_area / area2
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

        self.sample_list = [
            file
            for file in os.listdir(image_folder)
            if file.endswith((".jpg", ".jpeg", ".png", ".gif"))
        ]

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

        self.sample_list = [
            file
            for file in os.listdir(image_folder)
            if file.endswith((".jpg", ".jpeg", ".png", ".gif"))
        ]

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
    session = boto3.Session(profile_name="Developers_custom2_dev-pb-393436098818")

    # Create an S3 client using the session
    s3 = session.client("s3")
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

    print(
        f"{colors.GREEN}Uploaded {local_folder_path} to s3://{bucket_name}/{s3_folder_path}{colors.END}"
    )


def fraction_to_ratio(numerator, denominator):
    # Calculate the greatest common divisor (GCD)
    gcd = math.gcd(numerator, denominator)

    # Simplify the fraction
    simplified_numerator = numerator // gcd
    simplified_denominator = denominator // gcd

    # Return the simplified ratio as a string
    return f"{simplified_numerator}:{simplified_denominator}"


def get_avg_conf(conf_arr):
    # Create a dictionary to hold the sum of confidences and the count for each label
    label_confidence = {}

    for label, confidence in conf_arr:
        if label in label_confidence:
            label_confidence[label]["sum"] += confidence
            label_confidence[label]["count"] += 1
        else:
            label_confidence[label] = {"sum": confidence, "count": 1}

    # Calculate the average confidence for each label
    average_confidence = {
        label: values["sum"] / values["count"]
        for label, values in label_confidence.items()
    }
    return average_confidence


def get_label_from_sum_conf(conf_arr):
    # Create a dictionary to hold the sum of confidences and the count for each label
    label_confidence = {}

    for label, confidence in conf_arr:
        if label in label_confidence:
            label_confidence[label]["conf_sum"] += confidence
            label_confidence[label]["count"] += 1
        else:
            label_confidence[label] = {"conf_sum": confidence, "count": 1}

    max_conf = 0

    return_label = None

    for field in label_confidence:
        item = label_confidence[field]
        conf = item["conf_sum"]
        if max_conf < conf:
            max_conf = conf
            return_label = field

    return return_label


fields_inside_roi = {
    "Document_Info_block_pri": [
        "InvoiceDate",
        "InvoiceNumber",
        "TermsofPayment",
        "OrderNumber",
        "BuyerOrderDate",
        "ReferenceNumber",
    ],
    "Table_pri": [
        "TableActualQty",
        "TableBilledQty",
        "TableCGSTAmount",
        "TableDiscountAmount",
        "TableDiscountRate",
        "TableHSNSACCode",
        "TableIGSTAmount",
        "TableItemAmount",
        "TableItemBox",
        "TableItemRate",
        "TableItemRateUOM",
        "TableSGSTAmount",
    ],
    "Seller_address": [
        "SupplierAddress",
        "SupplierContactNo",
        "SupplierEmail",
        "SupplierGSTIN",
        "SupplierName",
        "SupplierPAN",
        "SupplierState",
    ],
    "Buyer_address": [
        "BuyerAddress",
        "BuyerContactNo",
        "BuyerEmail",
        "BuyerGSTIN",
        "BuyerName",
        "BuyerOrderDate",
        "BuyerPAN",
        "BuyerState",
    ],
    "Buyer_shipping": [
        "ConsigneeAddress",
        "ConsigneeContactNo",
        "ConsigneeEmail",
        "ConsigneeGSTIN",
        "ConsigneeName",
        "ConsigneePAN",
        "ConsigneeState",
    ],
    "Amount_details": [
        "LedgerDetailsLedgerAmount",
        "LedgerDetailsLedgerName",
        "LedgerDetailsLedgerRate",
        "SubAmount",
        "TotalAmount",
        # "TotalCGSTAmount",
        # "TotalCGSTRate",
        # "TotalDiscount",
        # "TotalIGSTAmount",
        # "TotalIGSTRate",
        # "TotalSGSTAmount",
        # "TotalSGSTRate",
        # "TotalTaxAmount",
        # "TotalTaxRate",
    ],
    "Total_amount": [
        "TotalAmount"
    ]
}
def biased_softmax(logits, best_roi_class, label2id, bias):
    if not best_roi_class or best_roi_class not in fields_inside_roi.keys():
        return F.softmax(logits, dim=0)

    
    interested_fields = fields_inside_roi[best_roi_class] + ["Other"]

    for field in interested_fields:
        interested_index = label2id[field]
        logits[interested_index] += bias

    softmax_output = F.softmax(logits, dim=0)

    return softmax_output


def remove_unnecessary_tokens(segment_tokens):
    start_index = 0
    words_to_ignore = ["", "no", "pan", "dated", "gstin", "gstinuin", "invoice", "bill", "date", "email", "mobile", "address", "name"]
    
    i = 1

    while i < len(segment_tokens):
        if segment_tokens[i][0] == 'Ġ':
            curr_word = tokenizer.convert_tokens_to_string(segment_tokens[start_index:i])
            # Keep only alphanumeric characters for comparison
            curr_word_alnum = re.sub(r'[^a-zA-Z0-9]', '', curr_word).lower()
            if curr_word_alnum in words_to_ignore:
                start_index = i

        i += 1

    return start_index

ledger_fields_list = ['TotalDiscount', "TotalSGSTRate", "TotalIGSTRate", 
                          "TotalTaxRate", "TotalCGSTAmount", "TotalSGSTAmount",
                          "TotalIGSTAmount", "TotalTaxAmount", "TotalCGSTRate"]

def convert_labels(tally_ai_json):
    
    
    Amount = []
    Rate = []


    for field in ledger_fields_list:
        if field in tally_ai_json:
            if isinstance(tally_ai_json[field],dict):
                if field.find("Amount") != -1 or field.find("Discount") != -1:
                    Amount.append(tally_ai_json[field].copy())
                elif field.find("Rate") != -1 :
                    Rate.append(tally_ai_json[field].copy())

            elif isinstance(tally_ai_json[field],list):
                if field.find("Amount") != -1 or field.find("Discount") != -1:
                    Amount.extend(tally_ai_json[field])
                elif field.find("Rate") != -1 :
                    Rate.extend(tally_ai_json[field])
   
            del tally_ai_json[field]

    max_index = max(len(Amount), len(Rate))
    
    for i in range(max_index):
        add_row = {}
        if i < len(Amount):
            add_row['LedgerAmount'] = Amount[i]
        if i < len(Rate):
            add_row['LedgerRate'] = Rate[i]
        
        if "LedgerDetails" in tally_ai_json:
            tally_ai_json['LedgerDetails'].append(add_row)
        else:
            tally_ai_json['LedgerDetails'] = [add_row]


    return tally_ai_json

def download_model_from_s3(job_dir):
    job_id = os.path.basename(job_dir)
    os.makedirs(job_dir, exist_ok=True)
    s3_client = boto3.client('s3')
    zip_filename = f"{job_id}.zip"
    zip_file_path = job_dir+".zip"
    s3_key = f"{model_name}/{zip_filename}"
    try:
        s3_client.download_file(model_artifacts_bucket, s3_key, zip_file_path)
        print(f"Downloaded {zip_file_path} from S3 bucket {model_artifacts_bucket}/{s3_key}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(job_dir)
        print(f"Extracted {zip_file_path} to {job_dir}")
        
        # Remove the zip file after extraction
        os.remove(zip_file_path)
        print(f"Deleted zip file {zip_file_path} after extraction")
    except FileNotFoundError:
        print(f"The file {zip_file_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def zip_directory(folder_path, zip_file_path):
    """Zip the contents of an entire directory."""
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    print(f"Directory {folder_path} zipped into {zip_file_path}")

def upload_file_to_s3(folder_path, model_name):
    """Upload a file to an S3 bucket."""
    s3_client = boto3.client('s3')

    zip_file_path = folder_path+".zip"

    zip_filename = os.path.basename(zip_file_path)

    zip_directory(folder_path, zip_file_path)

    s3_key = f"{model_name}/{zip_filename}"

    try:
        s3_client.upload_file(zip_file_path, model_artifacts_bucket, s3_key)
        print(f"File {zip_file_path} uploaded to {model_artifacts_bucket}/{s3_key}")

        os.remove(zip_file_path)
        print(f"File {zip_file_path} deleted after upload")

    except FileNotFoundError:
        print(f"The file {zip_file_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":

    processor = ProcessorWithEASYOCR()

    image_path = "AIBackend/DocAI/inpar-research/extractor/layoutLMv3/sample.jpg"

    pil_image = Image.open(image_path)

    response = processor.get_encodings(pil_image)
    print(response)
