import asyncio
import os
import sys
from pathlib import Path
import cv2
import img2pdf
from PIL import Image
import time
import json

# Add layoutLMv3 to the system path
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "layoutLMv3"))

# Import necessary classes and functions
from layoutLMv3.inference import layoutLMv3
from yolov5_table.modeling import yolov5_table
from yolov5_roi.modeling import yolov5_roi
from extractor_utils import (
    get_table_roi,
    extract_table_field,
    collate_non_table_fields_withroi,
)

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Initialize model instances
field_values_extractor = layoutLMv3()
roi_extractor = yolov5_roi()
table_structure_extractor = yolov5_table()


async def main(filename):
    pil_image = Image.open(filename)
    cv2_image = cv2.imread(filename)

    pdf_path = os.path.join(
        Path(
            r"D:\bitbucket\code.tallyai\AIBackend\DocAI\inpar-research\frontend\public"
        ),
        "document.pdf",
    )
    with open(pdf_path, "wb") as pdf_file, open(filename, "rb") as image_file:
        pdf_file.write(img2pdf.convert(image_file))
    print(f"PDF saved to: {pdf_path}")

    # Extract field values from the image
    field_values = await asyncio.to_thread(
        field_values_extractor.get_field_values, pil_image
    )
    # Extract ROIs from the image
    roi_result = await asyncio.to_thread(roi_extractor.get_roi, cv2_image)
    # Get table location from the ROI results
    table_ltwh = get_table_roi(roi_result, label=3)
    # Extract table structure from the image
    # table_row_col_wrt_page = await asyncio.to_thread(
    #     table_structure_extractor.get_table_structure, cv2_image, table_ltwh
    # )
    table_row_col_wrt_page = table_structure_extractor.get_table_structure(
        cv2_image, table_ltwh
    )

    # Process and combine field values and table structure
    tally_ai_json = extract_table_field(field_values, table_row_col_wrt_page)
    tally_ai_json = collate_non_table_fields_withroi(
        tally_ai_json, cv2_image, roi_result
    )
    print(tally_ai_json)

    with open(
        Path(
            r"D:\bitbucket\code.tallyai\AIBackend\DocAI\inpar-research\frontend\app\triage\components\sample.json"
        ),
        "w",
    ) as json_file:
        json.dump(tally_ai_json, json_file, indent=4)

    with open(os.path.join(current_dir, "sample_output.json"), "w") as json_file:
        json.dump(tally_ai_json, json_file, indent=4)


if __name__ == "__main__":
    # Run the main function asynchronously
    filename = os.path.join(current_dir, "sample.jpg")
    start_time = time.time()
    asyncio.run(main(filename))
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)