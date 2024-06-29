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
from layoutLMv3.inference_async import layoutLMv3
from yolov5_table.modeling import yolov5_table
from yolov5_roi.modeling_async import yolov5_roi
from extractor_utils import (
    get_table_roi,
    extract_table_field,
    collate_non_table_fields_withroi,
)

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# Input queues
pil_queue = asyncio.Queue()
cv2_queue = asyncio.Queue()

# Output queues
layout_queue = asyncio.Queue()
roi_queue = asyncio.Queue()

field_values_extractor = layoutLMv3(pil_queue, layout_queue)
roi_extractor = yolov5_roi(cv2_queue, roi_queue)
table_structure_extractor = yolov5_table()


async def read_image(filename):
    pil_image = Image.open(filename)
    cv2_image = cv2.imread(filename)
    return pil_image, cv2_image


async def main(filename):
    pil_image, cv2_image = await read_image(filename)
    await pil_queue.put(pil_image)
    await cv2_queue.put(cv2_image)

    layout_task = asyncio.create_task(field_values_extractor.get_field_values())
    roi_task = asyncio.create_task(roi_extractor.get_roi())

    await asyncio.gather(layout_task, roi_task)

    roi_result = await roi_queue.get()
    # lable 3 is for primary table
    table_ltwh = get_table_roi(roi_result, label=3)
    table_row_col_wrt_page = table_structure_extractor.get_table_structure(
        cv2_image, table_ltwh
    )

    field_values = await layout_queue.get()
    tally_ai_json = extract_table_field(field_values, table_row_col_wrt_page)
    tally_ai_json = collate_non_table_fields_withroi(
        tally_ai_json, cv2_image, roi_result
    )
    print(tally_ai_json)

    with open(os.path.join(current_dir, "sample_output.json"), "w") as json_file:
        json.dump(tally_ai_json, json_file, indent=4)


if __name__ == "__main__":
    # Run the main function asynchronously
    filename = os.path.join(current_dir, "sample.jpg")
    start_time = time.time()
    asyncio.run(main(filename))
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
