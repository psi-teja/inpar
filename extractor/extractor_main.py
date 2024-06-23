from yolov5_table.modeling import yolov5_table
from yolov5_roi.modeling import yolov5_roi
import sys, os

current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "layoutLMv3"))
from layoutLMv3.modeling import layoutLMv3
import traceback


import os, cv2, json
from PIL import Image
from extractor_utils import (
    get_table_roi,
    filling_tally_ai_json,
    cherry_picking
)
import time


field_values_extractor = layoutLMv3()
roi_extractor = yolov5_roi()
table_structure_extractor = yolov5_table()

os.environ["AWS_PROFILE"] = "Developers_custom1_tallydev-AI-381491826341"


def parse_invoice(pil_image, cv2_image):

    try:
        tally_ai_json_path = os.path.join(current_dir, "tally_ai.json")
        with open(tally_ai_json_path, "r") as f:
            tally_ai_json = json.load(f)

        roi_result = roi_extractor.get_roi(cv2_image)
        table_ltwh = get_table_roi(roi_result, label=3)
        amount_details_ltwh = get_table_roi(roi_result, label=5)
        table_row_wrt_page = table_structure_extractor.get_table_structure(
            cv2_image, table_ltwh
        )
        amount_details_row_wrt_page = table_structure_extractor.get_table_structure(
            cv2_image, amount_details_ltwh
        )

        (
            unique_bboxes,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
        ) = field_values_extractor.get_field_values(pil_image)

        filling_tally_ai_json(
            tally_ai_json,
            unique_bboxes,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
            table_row_wrt_page,
            amount_details_row_wrt_page,
        )

        cherry_picking(tally_ai_json)

        with open(os.path.join(current_dir, "output_tally_ai.json"), "w") as json_file:
            json.dump(tally_ai_json, json_file)

        return tally_ai_json
    except Exception as e:
        traceback.print_exc()
        print(e)
        return None


if __name__ == "__main__":
    filename = os.path.join(
        current_dir, "sample.jpg"
    )
    start_time = time.time()
    pil_image = Image.open(filename)
    cv2_image = cv2.imread(filename)
    tally_ai_json = parse_invoice(pil_image, cv2_image)
    print(tally_ai_json)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
