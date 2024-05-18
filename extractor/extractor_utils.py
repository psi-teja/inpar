import pandas as pd
from collections import Counter
import copy
from fitz import Rect


def get_table_roi(roi_result):
    max_conf = 0

    for result in roi_result:
        if int(result[-1]) == 6:

            conf = result[-2]

            if conf > max_conf:
                conf = max_conf
                table_ltwh = result[:4]
    return table_ltwh


def extract_table_field(field_values, table_row_col_wrt_page):

    column_boxes = table_row_col_wrt_page["column_boxes"]
    row_boxes = table_row_col_wrt_page["row_boxes"]

    pop_column_boxes = []
    for i in range(len(column_boxes) - 1):
        for j in range(i + 1, len(column_boxes)):
            inter_area = iob(column_boxes[i]["bbox"], column_boxes[j]["bbox"])
            if inter_area >= 0.5:
                pop_column_boxes.append(column_boxes[j])
    new_column_boxes = [x for x in column_boxes if x not in pop_column_boxes]

    pop_row_boxes = []
    for i in range(len(row_boxes) - 1):
        for j in range(i + 1, len(row_boxes)):
            inter_area = iob(row_boxes[i]["bbox"], row_boxes[j]["bbox"])
            if inter_area >= 0.5:
                pop_row_boxes.append(row_boxes[j])
    new_row_boxes = [x for x in row_boxes if x not in pop_row_boxes]

    cells = []
    for row_box in new_row_boxes:
        for column_box in new_column_boxes:
            cells.append(
                {
                    "bbox": bbox_overlap(row_box["bbox"], column_box["bbox"]),
                    "score": (row_box["score"] + column_box["score"]) / 2,
                }
            )

    tally_ai_json = {}
    sample_table = {}
    for key, value in field_values.items():
        if key[:5] == "Table":
            sample_table[key[5:]] = value
        else:
            tally_ai_json[key] = value

    arranged_cells = sort_table_cells_main(cells, sample_table)
    table_json = arranged_cell_finaljson(arranged_cells)

    tally_ai_json["Table"] = table_json
    return tally_ai_json


def iob(bbox1, bbox2):
    intersection = Rect(bbox1).intersect(bbox2)
    bbox1_area = Rect(bbox1).get_area()  # .getArea()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area  # getArea()
    return 0


def calculate_ioa(bbox1, bbox2):
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    # Check if there's no intersection (one or both rectangles have zero area)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # Calculate area of each rectangle
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    if not area1 or not area2:
        return 0
    if area1 < area2:
        ioa = intersection_area / area1
    else:
        ioa = intersection_area / area2
    return ioa


def sort_table_cells(df):
    num_row = len(df)
    # sort and segment line-wise
    sorted_df = df.sort_values(by="ymin")
    arranged_cells = []
    start_row = 0
    row = 0
    while start_row < num_row:
        ymin = sorted_df.iloc[start_row]["ymin"]
        ymax = sorted_df.iloc[start_row]["ymax"]
        y_thresh = ymin + (ymax - ymin) / 2
        while row < num_row and ymin <= y_thresh:
            row += 1
            if row < num_row:
                ymin = sorted_df.iloc[row]["ymin"]
            else:
                break
        end_row = copy.deepcopy(row)
        block = sorted_df.iloc[start_row:end_row].sort_values(by="xmin")
        cell_row_box = fetch_row_frm_block(block)
        arranged_cells.append(cell_row_box)
        start_row = copy.deepcopy(end_row)
    return arranged_cells


def fetch_row_frm_block(block):
    num_row = len(block.index)
    cell_row_box = []
    for row in range(num_row):
        cell = {}
        temp = block.iloc[row]
        cell["bbox"] = [
            temp["xmin"],
            temp["ymin"],
            temp["xmax"],
            temp["ymax"],
        ]
        cell["score"] = temp["score"]
        cell["span"] = temp["span"]
        cell_row_box.append(cell)
    return cell_row_box


def bbox_overlap(bbox1, bbox2):
    # Extract coordinates of the bounding boxes
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    # Calculate the coordinates of the intersection rectangle
    xmin_intersection = max(xmin1, xmin2)
    ymin_intersection = max(ymin1, ymin2)
    xmax_intersection = min(xmax1, xmax2)
    ymax_intersection = min(ymax1, ymax2)

    return [xmin_intersection, ymin_intersection, xmax_intersection, ymax_intersection]


def sort_table_cells_main(cells, field_values):
    cells_span = []
    for cell1 in cells:
        temp1 = []
        for key1, value1 in field_values.items():
            for val1 in value1:
                val1_box = [
                    val1["location"]["ltwh"][0],
                    val1["location"]["ltwh"][1],
                    val1["location"]["ltwh"][0] + val1["location"]["ltwh"][2],
                    val1["location"]["ltwh"][1] + val1["location"]["ltwh"][3],
                ]
                inter_area = iob(val1_box, cell1["bbox"])
                if (
                    inter_area >= 0.3
                    and cell1["bbox"][0] < val1_box[0] < cell1["bbox"][2]
                    and cell1["bbox"][1] < val1_box[1] < cell1["bbox"][3]
                ):
                    temp1.append({key1: val1})
        cell1["span"] = temp1
        cells_span.append(cell1)

    cells_data = []
    for cell in cells_span:
        cell_d = {}
        cell_d["xmin"], cell_d["ymin"], cell_d["xmax"], cell_d["ymax"] = cell["bbox"]
        cell_d["score"] = cell["score"]
        cell_d["span"] = cell["span"]
        cells_data.append(cell_d)
    df = pd.DataFrame(cells_data)
    arranged_cells = sort_table_cells(df)
    return arranged_cells


def arranged_cell_finaljson(arranged_cells):
    data_rows = []
    for i, row in enumerate(arranged_cells):
        rows = []
        for j, cellr in enumerate(row):
            spans = cellr["span"]
            if len(spans) == 0:
                continue
            column_lst = [list(item.keys())[0] for item in spans]
            counts = Counter(column_lst)
            most_common = counts.most_common(1)[0][0]
            indices = [
                index for index, value in enumerate(column_lst) if value == most_common
            ]
            new_spans = [spans[indc] for indc in indices]
            text = ""
            for span in new_spans:
                if "text" in span[most_common]:
                    text = text + " " + span[most_common]["text"]
            cellr["header"] = most_common
            cellr["text"] = text.strip()
            rows.append(cellr)
        if len(rows) == 0:
            continue
        data_rows.append(rows)

    table_ta = []
    for row_box in data_rows:
        box_j = {}
        for box in row_box:
            header = box["header"]
            box_j[header] = {}
            box_j[header]["text"] = box["text"]
            box_j[header]["location"] = {}
            box_j[header]["location"]["ltwh"] = [
                box["bbox"][0],
                box["bbox"][1],
                box["bbox"][2] - box["bbox"][0],
                box["bbox"][3] - box["bbox"][1],
            ]
            box_j[header]["location"]["pageNo"] = 1  # needs to add from layoutlm model
        table_ta.append(box_j)
    return table_ta


def collate_non_table_fields(tally_ai_json):
    for field in tally_ai_json:
        if field == "Table":
            continue

        extractedValues = tally_ai_json[field]

        text = ""
        ltwh = None
        pageNo = 1

        for value in extractedValues:
            text += " " + value["text"]

            if not ltwh:
                ltwh = value["location"]["ltwh"]
            else:
                l1, t1, w1, h1 = value["location"]["ltwh"]
                l2, t2, w2, h2 = ltwh

                l = min(l1, l2)
                t = min(t1, t2)
                w = max(l1 + w1, l2 + w2) - l
                h = max(t1 + h1, t2 + h2) - t

                ltwh = [l,t,w,h]

        tally_ai_json[field] = {"text": text.strip(), "location": {"ltwh": ltwh, "pageNo": pageNo}}

    
    return tally_ai_json