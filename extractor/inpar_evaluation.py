import os, random
import json
from collections import defaultdict
from typing import List, Dict, Tuple
from extractor_main import parse_invoice
import openpyxl
from PIL import Image
import cv2, re
from layoutLMv3.layoutLMv3_utils import calculate_iou
from fuzzywuzzy import fuzz
from tqdm import tqdm

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

current_dir = os.path.dirname(__file__)


def load_json(file_path: str) -> Dict:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return False

def clean_text(text):
    # Remove all types of whitespace characters (spaces, tabs, newlines) and convert to lowercase
    return re.sub(r'\s+', '', text).lower()

def get_model_predictions(image_path):
    # Process image
    pil_image = Image.open(image_path)
    cv2_image = cv2.imread(image_path)

    tally_ai_json = parse_invoice(pil_image, cv2_image, pageNo=1)

    return tally_ai_json

def evaluate_model(images_folder: str, gt_folder: str, iou_threshold: float = 0.3, fuzzy_threshold: float = 70) -> Dict[str, Dict[str, float]]:
    """Evaluate the model performance field-wise."""
    field_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
    
    image_files = os.listdir(images_folder)

    random.shuffle(image_files)
    
    # assert set(image_files) == set(gt_files), "Prediction and ground truth files do not match"
    number_of_samples = 2000

    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Metrics"
    worksheet.append(["Filename", "Field", "Pred Text", "GT text", "Confusion", "Fuzzy Ratio", "IOU Score"])


    for filename in tqdm(image_files[:number_of_samples]):
        image_path = os.path.join(images_folder, filename)
        gt_path = os.path.join(gt_folder, filename.split('.')[0]+".json")
        
        predictions = get_model_predictions(image_path)
        ground_truth = load_json(gt_path)
        
        if not ground_truth:
            continue

        for field in ground_truth.keys():
            if isinstance(ground_truth[field], dict):
                gt_text = clean_text(ground_truth[field]['text'])
                l1, t1, w1, h1 = ground_truth[field]['location']['ltwh']

                pred_text = clean_text(predictions[field]['text'])
                l2, t2, w2, h2 = predictions[field]['location']['ltwh']

                fuzzy_ratio = fuzz.ratio(gt_text, pred_text)
                iou_score = calculate_iou(l1, t1, w1, h1, l2, t2, w2, h2)

                confusion = None

                if pred_text == "":
                    field_metrics[field]['fn'] += 1
                    confusion = "FN"
                elif fuzzy_ratio >= fuzzy_threshold and iou_score >= iou_threshold:
                    field_metrics[field]['tp'] += 1
                    confusion = "TP"
                else:
                    field_metrics[field]['fp'] += 1
                    confusion = "FP"
                    
                field_metrics[field]['support'] += 1
                
                # Write to the XLSX file
                worksheet.append([filename, field, pred_text, gt_text, confusion , fuzzy_ratio, iou_score])
            elif field == 'Table':
                gt_dict = {}
                pred_dict = {}
                for row in ground_truth[field]:
                    for col in row:
                        if not isinstance(row[col], dict) or col == "TaxAmount": continue
                        text = clean_text(row[col]['text'])
                        ltwh = row[col]['location']['ltwh']
                        if col not in gt_dict:
                            gt_dict[col] = []
                        gt_dict[col].append((text, ltwh))

                for row in predictions[field]:   
                    for col in row: 
                        text = clean_text(row[col]['text'])
                        ltwh = row[col]['location']['ltwh']
                        if col not in pred_dict:
                            pred_dict[col] = []
                        pred_dict[col].append((text, ltwh))

                for col in gt_dict:
                    for i in range(len(gt_dict[col])):

                        gt_text = clean_text(gt_dict[col][i][0])
                        l1, t1, w1, h1 = gt_dict[col][i][1]

                        if col in pred_dict.keys() and i < len(pred_dict[col]): 
                            pred_text = clean_text(pred_dict[col][i][0])
                            l2, t2, w2, h2 = pred_dict[col][i][1]
                        else:
                            pred_text = ""
                            l2, t2, w2, h2 = [0,0,0,0]
                        
                            
                        

                        fuzzy_ratio = fuzz.ratio(gt_text, pred_text)
                        iou_score = calculate_iou(l1, t1, w1, h1, l2, t2, w2, h2)

                        confusion = None

                        if pred_text == "":
                            field_metrics[field+col]['fn'] += 1
                            confusion = "FN"
                        elif fuzzy_ratio >= fuzzy_threshold and iou_score >= iou_threshold:
                            field_metrics[field+col]['tp'] += 1
                            confusion = "TP"
                        else:
                            field_metrics[field+col]['fp'] += 1
                            confusion = "FP"
                            
                        field_metrics[field+col]['support'] += 1
                    
                        # Write to the XLSX file
                        worksheet.append([filename, field+col, pred_text, gt_text, confusion , fuzzy_ratio, iou_score])



    # Save the workbook
    workbook.save(os.path.join(current_dir,"metrics.xlsx"))       

    
    field_wise_metrics = {}
    for field, metrics in field_metrics.items():
        tp = metrics['tp']
        fp = metrics['fp']
        fn = metrics['fn']
        support = metrics['support']
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        field_wise_metrics[field] = {
            'precision': precision,
            'recall': recall,
            'support': support,
            'f1_score': f1_score
        }
    
    return field_wise_metrics

def write_metrics_to_xlsx(metrics: Dict[str, Dict[str, float]], output_path: str):
    """Write evaluation metrics to an XLSX file."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation Metrics"

    # Write headers
    ws.append(["Field", "Precision", "Recall", "Support", "F1 Score"])

    # Write metrics
    for field, metric in metrics.items():
        ws.append([
            field,
            metric['precision'],
            metric['recall'],
            metric['support'],
            metric['f1_score']
        ])

    # Save the workbook
    wb.save(output_path)

if __name__ == '__main__':


    # Define paths to prediction and ground truth folders
    images_folder = "AIBackend/DocAI/inpar-research/extractor/layoutLMv3/datasets/imerit/phase1/images"
    gt_folder = "AIBackend/DocAI/inpar-research/extractor/layoutLMv3/datasets/imerit/phase1/tally_ai_jsons"

    # Evaluate the model
    field_wise_metrics = evaluate_model(images_folder, gt_folder)

    # Define the output path for the XLSX file
    output_xlsx_path = os.path.join(current_dir, 'evaluation_metrics.xlsx')

    # Write the metrics to an XLSX file
    write_metrics_to_xlsx(field_wise_metrics, output_xlsx_path)

    print(f"Evaluation metrics written to {output_xlsx_path}")
