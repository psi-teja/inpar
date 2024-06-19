import pandas as pd
from collections import Counter
import copy
from fitz import Rect
import cv2, os, json
current_dir = os.path.dirname(__file__)

def get_table_roi(roi_result, label):
    max_conf = 0
    table_ltwh = []
    for result in roi_result:
        if int(result[-1]) == label:

            conf = result[-2]

            if conf > max_conf:
                max_conf = conf
                table_ltwh = result[:4]
    return table_ltwh


def draw_roi_results_image(roi_result, cv2_image):
    for i in range(len(roi_result)):
        ltwh = roi_result[i, :]
        l, t, w, h = ltwh[:4]
        height, width, _ = cv2_image.shape
        x1 = max(0, int((l - w / 2 - 0.001) * width))
        y1 = max(0, int((t - h / 2 - 0.001) * height))
        x2 = min(width, int((l + w / 2 + 0.001) * width))
        y2 = min(height, int((t + h / 2 + 0.001) * height))
        cv2.rectangle(cv2_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            cv2_image,
            str(ltwh[-1]),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )
    cv2.imwrite("output_image.jpg", cv2_image)


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
            x_min = float("inf")
            y_min = float("inf")
            x_max = float("-inf")
            y_max = float("-inf")
            for span in new_spans:
                if "text" in span[most_common]:
                    text = text + " " + span[most_common]["text"]
                    bbox_0 = span[most_common]["location"]["ltwh"][0]
                    bbox_1 = span[most_common]["location"]["ltwh"][1]
                    bbox_2 = (
                        span[most_common]["location"]["ltwh"][0]
                        + span[most_common]["location"]["ltwh"][2]
                    )
                    bbox_3 = (
                        span[most_common]["location"]["ltwh"][1]
                        + span[most_common]["location"]["ltwh"][3]
                    )
                    x_min = min(x_min, bbox_0)
                    y_min = min(y_min, bbox_1)
                    x_max = max(x_max, bbox_2)
                    y_max = max(y_max, bbox_3)
            cellr["header"] = most_common
            cellr["text"] = text.strip()
            cellr["words_bbox"] = [x_min, y_min, x_max, y_max]
            rows.append(cellr)
        if len(rows) == 0:
            continue
        data_rows.append(rows)

    table_ta = []
    for row_box in data_rows:
        box_j = {}
        for ind, box in enumerate(row_box):
            header = box["header"]
            if header in box_j:
                header = header + "_" + str(ind)
            else:
                header = header
            box_j[header] = {}
            box_j[header]["text"] = box["text"]
            box_j[header]["location"] = {}
            box_j[header]["location"]["ltwh"] = [
                box["words_bbox"][0],
                box["words_bbox"][1],
                box["words_bbox"][2] - box["words_bbox"][0],
                box["words_bbox"][3] - box["words_bbox"][1],
            ]
            box_j[header]["location"]["pageNo"] = 1  # needs to add from layoutlm model
        table_ta.append(box_j)
    return table_ta


def extract_ledger_details(field_values):
    tally_ai_json = {}
    ledger_details = {}
    for key, value in field_values.items():
        if len(key) > 13:
            if key[:13] == "LedgerDetails":
                ledger_details[key[13:]] = value
            else:
                tally_ai_json[key] = value
        else:
            tally_ai_json[key] = value

    if len(ledger_details) != 0:
        ledger_index = True
        if "LedgerName" in ledger_details:
            ledger_name = ledger_details["LedgerName"]
        else:
            ledger_name = []
        if "LedgerRate" in ledger_details:
            ledger_rate = ledger_details["LedgerRate"]
        else:
            ledger_rate = []
        if "LedgerAmount" in ledger_details:
            ledger_amount = ledger_details["LedgerAmount"]
        else:
            ledger_amount = []
    else:
        return tally_ai_json

    ledger_tab = []
    if ledger_index:
        ledger_dict = {}
        for i in range(max(len(ledger_amount), len(ledger_name), len(ledger_rate))):
            ledger_dict = {}
            if i < len(ledger_name):
                ledger_dict["LedgerName"] = ledger_name[i]
            if i < len(ledger_rate):
                ledger_dict["LedgerRate"] = ledger_rate[i]
            if i < len(ledger_amount):
                ledger_dict["LedgerAmount"] = ledger_amount[i]
            ledger_tab.append(ledger_dict)
    tally_ai_json["LedgerDetails"] = ledger_tab
    return tally_ai_json


def collate_non_table_fields(tally_ai_json):

    for field in tally_ai_json:
        if field == "Table":
            continue

        if field == "LedgerDetails":
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

                ltwh = [l, t, w, h]

        tally_ai_json[field] = {
            "text": text.strip(),
            "location": {"ltwh": ltwh, "pageNo": pageNo},
        }

    return tally_ai_json


def collate_fieldwise(extractedValues, bbox):
    if bbox is None:
        extractedValues2 = extractedValues
    else:
        extractedValues2 = []
        for value in extractedValues:
            val1_box = [
                value["location"]["ltwh"][0],
                value["location"]["ltwh"][1],
                value["location"]["ltwh"][0] + value["location"]["ltwh"][2],
                value["location"]["ltwh"][1] + value["location"]["ltwh"][3],
            ]
            if iob(val1_box, bbox) >= 0.8:
                extractedValues2.append(value)

    text = ""
    ltwh = None
    pageNo = 1
    for val in extractedValues2:
        text += " " + val["text"]
        if not ltwh:
            ltwh = val["location"]["ltwh"]
        else:
            l1, t1, w1, h1 = val["location"]["ltwh"]
            l2, t2, w2, h2 = ltwh
            l = min(l1, l2)
            t = min(t1, t2)
            w = max(l1 + w1, l2 + w2) - l
            h = max(t1 + h1, t2 + h2) - t
            ltwh = [l, t, w, h]
    location = {"ltwh": ltwh, "pageNo": pageNo}
    return text.strip(), location


def collate_non_table_fields_withroi(tally_ai_json, cv2_image, roi_result):

    height, width, _ = cv2_image.shape
    roi_extracted = [int(result[-1]) for result in roi_result]
    if 0 in roi_extracted:
        buyer_ltwh = get_table_roi(roi_result, label=0)
        l, t, w, h = buyer_ltwh
        x1 = max(0, int((l - w / 2 - 0.001) * width))
        y1 = max(0, int((t - h / 2 - 0.001) * height))
        x2 = min(width, int((l + w / 2 + 0.001) * width))
        y2 = min(height, int((t + h / 2 + 0.001) * height))
        buyer_ltwh = [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]
    else:
        buyer_ltwh = None

    if 8 in roi_extracted:
        consignee_ltwh = get_table_roi(roi_result, label=8)
        l, t, w, h = consignee_ltwh
        x1 = max(0, int((l - w / 2 - 0.001) * width))
        y1 = max(0, int((t - h / 2 - 0.001) * height))
        x2 = min(width, int((l + w / 2 + 0.001) * width))
        y2 = min(height, int((t + h / 2 + 0.001) * height))
        consignee_ltwh = [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]
    else:
        consignee_ltwh = None

    if 2 in roi_extracted:
        document_Info_block_pri_ltwh = get_table_roi(roi_result, label=2)
        l, t, w, h = document_Info_block_pri_ltwh
        x1 = max(0, int((l - w / 2 - 0.001) * width))
        y1 = max(0, int((t - h / 2 - 0.001) * height))
        x2 = min(width, int((l + w / 2 + 0.001) * width))
        y2 = min(height, int((t + h / 2 + 0.001) * height))
        document_Info_block_pri_ltwh = [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]
    else:
        document_Info_block_pri_ltwh = None

    for field in tally_ai_json:
        if field == "Table":
            continue

        if field == "LedgerDetails":
            continue

        if field == "BuyerAddress":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerState":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerName":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerPAN":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerContactNo":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerEmail":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "BuyerGSTIN":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, buyer_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue

        if field == "ConsigneeAddress":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneeState":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneeName":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneeEmail":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneeContactNo":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneePAN":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ConsigneeGSTIN":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(extractedValues, consignee_ltwh)
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue

        if field == "InvoiceNumber":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "InvoiceDate":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "TermsofPayment":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "DispatchThrough":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "OrderNumber":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "ReferenceNumber":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "OrderDueDate":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "Destination":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
            continue
        if field == "OtherReference":
            extractedValues = tally_ai_json[field]
            text, location = collate_fieldwise(
                extractedValues, document_Info_block_pri_ltwh
            )
            tally_ai_json[field] = {
                "text": text,
                "location": location,
            }
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
                ltwh = [l, t, w, h]

        tally_ai_json[field] = {
            "text": text.strip(),
            "location": {"ltwh": ltwh, "pageNo": pageNo},
        }

    return tally_ai_json


def min_super_rectangle(ltwh1, ltwh2):
    l1, t1, w1, h1 = ltwh1
    l2, t2, w2, h2 = ltwh2

    l = min(l1, l2)
    t = min(t1, t2)
    w = max(l1 + w1, l2 + w2) - l
    h = max(t1 + h1, t2 + h2) - t

    return [l, t, w, h]


def find_items_table_row(table_row_wrt_page, word_yc):

    for i in range(len(table_row_wrt_page)):
        x1, y1, x2, y2 = table_row_wrt_page[i]["bbox"]

        if word_yc > y1 and word_yc < y2:
            return i - 1

    return -1


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


def get_shortest_dist_btw_rect(ltwh1, ltwh2):
    # Extract coordinates and dimensions
    x1, y1, w1, h1 = ltwh1
    x2, y2, w2, h2 = ltwh2

    # Calculate the right and bottom coordinates
    right1, bottom1 = x1 + w1, y1 + h1
    right2, bottom2 = x2 + w2, y2 + h2

    # Calculate the horizontal and vertical distances
    dx = max(0, x2 - right1, x1 - right2)
    dy = max(0, y2 - bottom1, y1 - bottom2)

    # Calculate the shortest distance
    distance = (dx**2 + dy**2) ** 0.5

    return distance

def cherry_picking(tally_ai_json):
    for field in tally_ai_json:
        if field.find("Table") == -1 and field.find("LedgerDetails")==-1:
            if isinstance(tally_ai_json[field], list):
                best_extractedValue = None
                best_conf = 0
                for i in range(len(tally_ai_json[field])):
                    extractedValue = tally_ai_json[field][i]
                    conf_arr = extractedValue['conf_arr']
                    conf = get_avg_conf(conf_arr)[field]
                    if conf > best_conf:
                        best_conf = conf
                        best_extractedValue = extractedValue
            
                tally_ai_json[field] = best_extractedValue

    return tally_ai_json


def filling_tally_ai_json(
    tally_ai_json,
    unique_bboxes_xyxy,
    words_in_unique_bboxes,
    labels_for_unique_bboxes,
    conf_arr_of_unique_bboxes,
    table_row_wrt_page,
    amount_details_row_wrt_page,
):

    if table_row_wrt_page:
        table_row_wrt_page = sorted(table_row_wrt_page, key=lambda item: item["bbox"][1])

    num_of_rows = len(table_row_wrt_page) - 1

    row_template = tally_ai_json["Table"][0]

    tally_ai_json["Table"] = [dict(row_template) for _ in range(num_of_rows)]

    for bbox, words, label, conf_arr in zip(
        unique_bboxes_xyxy,
        words_in_unique_bboxes,
        labels_for_unique_bboxes,
        conf_arr_of_unique_bboxes,
    ):

        x1, y1, x2, y2 = bbox

        avg_conf = get_avg_conf(conf_arr)

        l1 = x1
        t1 = y1
        w1 = x2 - x1
        h1 = y2 - y1

        word_yc = (y1 + y2) / 2

        ltwh1 = [l1, t1, w1, h1]

        if label == "Other":
            continue

        if label.find("GSTIN") != -1 and words.find("GSTIN") != -1:
            continue

        if label.find("Table") != -1:
            if not table_row_wrt_page: continue
            row_index = find_items_table_row(table_row_wrt_page, word_yc)
            col_name = label[5:]
            extractedValue = tally_ai_json["Table"][row_index][col_name]

            if extractedValue["location"]["pageNo"] == 0:
                tally_ai_json["Table"][row_index][col_name] = {
                    "text": words,
                    "location": {
                        "pageNo": 1,
                        "ltwh": ltwh1,
                    },
                }
            else:
                new_text = extractedValue["text"] + " " + words
                old_ltwh = extractedValue["location"]["ltwh"]
                new_ltwh = min_super_rectangle(old_ltwh, ltwh1)
                tally_ai_json["Table"][row_index][col_name] = {
                    "text": new_text,
                    "location": {
                        "pageNo": 1,
                        "ltwh": new_ltwh,
                    },
                }

        elif label.find("LedgerDetails") != -1:
            col_name = label[13:]
            tally_ai_json["LedgerDetails"][0][col_name] = {
                "text": words,
                "location": {
                    "pageNo": 1,
                    "ltwh": ltwh1,
                },
            }
        else:
            extractedValue = tally_ai_json[label]
            if isinstance(extractedValue, dict):
                if extractedValue["location"]["pageNo"] == 0:
                    tally_ai_json[label] = {
                        "text": words,
                        "location": {
                            "pageNo": 1,
                            "ltwh": ltwh1
                        },
                        "conf_arr": conf_arr
                    }
                else:
                    new_text = extractedValue["text"] + " " + words
                    old_ltwh = extractedValue["location"]["ltwh"]
                    old_conf_arr = extractedValue['conf_arr']
                    shortest_dist = get_shortest_dist_btw_rect(ltwh1, old_ltwh)
                    if shortest_dist < 0.025:
                        new_ltwh = min_super_rectangle(old_ltwh, ltwh1)
                        tally_ai_json[label] = {
                            "text": new_text,
                            "location": {"pageNo": 1, "ltwh": new_ltwh},
                            "conf_arr": old_conf_arr+conf_arr
                        }
                    else:
                        tally_ai_json[label] = [extractedValue]
                        tally_ai_json[label].append(
                            {
                                "text": words,
                                "location": {"pageNo": 1, "ltwh": ltwh1},
                                "conf_arr": old_conf_arr+conf_arr
                            }
                        )

            else:
                min_shortest_dist = float('inf')
                group_index = None
                for i in range(len(tally_ai_json[label])):
                    extractedValue = tally_ai_json[label][i]
                    old_ltwh = extractedValue['location']['ltwh']
                    shortest_dist = get_shortest_dist_btw_rect(ltwh1, old_ltwh)
                    if shortest_dist < min_shortest_dist:
                        min_shortest_dist = shortest_dist
                        group_index = i

                if min_shortest_dist < 0.025:
                    extractedValue = tally_ai_json[label][group_index]
                    new_text = extractedValue["text"] + " " + words
                    old_ltwh = extractedValue["location"]["ltwh"]
                    old_conf_arr = extractedValue['conf_arr']
                    new_ltwh = min_super_rectangle(old_ltwh,ltwh1)
                    tally_ai_json[label][group_index] = {
                        "text": new_text,
                        "location": {"pageNo": 1, "ltwh": new_ltwh},
                        'conf_arr': old_conf_arr+conf_arr
                    }
                else:
                    tally_ai_json[label].append(
                            {
                                "text": words,
                                "location": {"pageNo": 1, "ltwh": ltwh1},
                                'conf_arr': conf_arr
                            }
                        )

    with open(os.path.join(current_dir, "output_pre_label.json"), "w") as json_file:
        json.dump(tally_ai_json, json_file)

    return tally_ai_json


if __name__ == "__main__":
    ltwh1 = [0, 0, 0.5, 0.5]
    ltwh2 = [0.6, 0.6, 0.5, 0.5]
    distance = get_shortest_dist_btw_rect(ltwh1, ltwh2)
    print(distance)
