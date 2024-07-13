import sys, os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "layoutLMv3"))
sys.path.append(os.path.join(current_dir, "yolov5_table"))
sys.path.append(os.path.join(current_dir, "yolov5_roi"))
from layoutLMv3.inference import layoutLMv3
from yolov5_table.inference import yolov5_table
from yolov5_roi.inference import yolov5_roi
import traceback
import copy

import os, cv2, json
from PIL import Image
from extractor_utils import get_table_roi, filling_tally_ai_json, cherry_picking
import time

import pathlib

if os.name == 'nt':  # Windows OS
    pathlib.PosixPath = pathlib.Path

field_values_extractor = layoutLMv3()
roi_extractor = yolov5_roi()
table_structure_extractor = yolov5_table()


def parse_invoice(pil_image, cv2_image, pageNo):

    # print(f"Processing Page No: {pageNo}")

    try:
        tally_ai_json_path = os.path.join(current_dir, "dummy.json")
        with open(tally_ai_json_path, "r") as f:
            tally_ai_json = json.load(f)

        roi_result = roi_extractor.get_roi(cv2_image)
        table_XcYcWH = get_table_roi(roi_result, label_name="Table_pri")

        (
            unique_bboxes,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
        ) = field_values_extractor.get_field_values(pil_image, roi_data=roi_result)

        table_structure_extractor.get_table_structure(
            cv2_image,
            table_XcYcWH,
            tally_ai_json,
            unique_bboxes,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
            pageNo,
        )

        filling_tally_ai_json(
            tally_ai_json,
            unique_bboxes,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
            {},
            pageNo,
        )

        cherry_picking(tally_ai_json)

        return tally_ai_json
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {}


def merge_with_final_output(tally_ai_json_page, tally_ai_json):
    if not tally_ai_json:
        tally_ai_json = copy.deepcopy(tally_ai_json_page)
        return tally_ai_json

    if "Table" in tally_ai_json_page and "Table" in tally_ai_json:
        tally_ai_json["Table"].extend(tally_ai_json_page["Table"])
    elif "Table" in tally_ai_json_page:
        tally_ai_json["Table"] = tally_ai_json_page["Table"]

    return tally_ai_json


if __name__ == "__main__":
    # file_path = os.path.join(current_dir, "sample.jpg")

    # file_path = "/home/saiteja/DocAI/code.tallyai/AIBackend/DocAI/inpar-research/extractor/layoutLMv3/datasets/imerit/images/1714657328228d58083ca-0cc7-4f60-a7b3-877066ac3b9e_page2.jpeg"
    file_path = "C:/Users/sathish.v/Desktop/1714655287929ea2da5f4-7236-4788-9d7c-a64703bf342a_page1.jpeg"
    start_time = time.time()
    pil_image = Image.open(file_path)
    cv2_image = cv2.imread(file_path)
    tally_ai_json = parse_invoice(pil_image, cv2_image, pageNo=1)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
