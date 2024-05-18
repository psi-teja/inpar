from yolov5_table.modeling import yolov5_table
from yolov5_roi.modeling import yolov5_roi
import sys, os
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "layoutLMv3"))
from layoutLMv3.inference import layoutLMv3

import os, cv2, json
from PIL import Image
from extractor_utils import get_table_roi, extract_table_field, collate_non_table_fields
import img2pdf



field_values_extractor = layoutLMv3()
roi_extractor = yolov5_roi()
table_structure_extractor = yolov5_table()


filename = os.path.join(current_dir, 'd:/Tally Solutions Pvt. Ltd/Tally-AI - DocAI/images_and_gt_json/GST/595.jpeg')

pil_image = Image.open(filename)
cv2_image = cv2.imread(filename)

pdf_path = os.path.join("D:/tally/bitBucket/code.tallyai/AIBackend/DocAI/inpar-research/frontend/public", "document.pdf")
with open(pdf_path, "wb") as pdf_file, open(filename, "rb") as image_file:
        pdf_file.write(img2pdf.convert(image_file))

print(f"PDF saved to: {pdf_path}")

field_values = field_values_extractor.get_field_values(pil_image)

roi_result = roi_extractor.get_roi(cv2_image)

table_ltwh =  get_table_roi(roi_result)

table_row_col_wrt_page = table_structure_extractor.get_table_structure(cv2_image, table_ltwh)

tally_ai_json = extract_table_field(field_values, table_row_col_wrt_page)

collate_non_table_fields(tally_ai_json)

print(tally_ai_json)

with open("D:/tally/bitBucket/code.tallyai/AIBackend/DocAI/inpar-research/frontend/app/triage/components/sample.json", 'w') as json_file:
    json.dump(tally_ai_json, json_file, indent=4)




