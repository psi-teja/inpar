# %%
import os
import json
import cv2
from tqdm import tqdm

# %%
current_dir = os.path.dirname(__file__)

roi_labels = os.path.join(current_dir, "yolov5_roi", "datasets", "imerit", "labels")

layout_images = os.path.join(current_dir, "layoutLMv3", "datasets", "imerit", "images")
individual_fields_labels = os.path.join(
    current_dir, "layoutLMv3", "datasets", "imerit", "tally_ai_jsons"
)

list_of_images = os.listdir(layout_images)


# %%
dataset_folder = os.path.join(current_dir, "yolov5_table", "datasets", "imerit")
images_folder = os.path.join(dataset_folder, "images")
labels_folder = os.path.join(dataset_folder, "labels")

# Create directories if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

number_of_samples = 0

# %%

for filename in tqdm(list_of_images):

    individualfield_json_path = os.path.join(individual_fields_labels, filename.split(".")[0] + ".json")
    roi_txt_path = os.path.join(roi_labels, filename.split(".")[0] + ".txt")

    output_file_path = os.path.join(labels_folder, filename.split(".")[0] + ".txt")

    # if os.path.exists(output_file_path):
    #     number_of_samples += 1
    #     continue

    if not os.path.exists(individual_fields_labels) or not os.path.exists(roi_txt_path):
        continue

    number_of_samples += 1
   
    with open(individualfield_json_path, "r") as file:
        individual_field_json = json.load(file)

    if "Table" not in individual_field_json.keys():
        continue

    table_json = individual_field_json["Table"]
    rows_xyxy = []
    columns = {}
    for item in table_json:
        row = []
        for key, value in item.items():
            if key in ["object_group_id", "BatchGodownDetails", "ItemName", "ItemDescription", 'ItemRateUOM']:
                continue
            
            if "location" in value.keys():
                row.append(value["location"]["ltwh"])
                if key in columns:
                    columns[key].append(value["location"]["ltwh"])
                else:
                    columns[key] = []
                    columns[key].append(value["location"]["ltwh"])
            else:
                for key2, value2 in value.items():
                    row.append(value2["location"]["ltwh"])
                    if key2 in columns:
                        columns[key2].append(value2["location"]["ltwh"])
                    else:
                        columns[key2] = []
                        columns[key2].append(value2["location"]["ltwh"])
        if row:
            min_x = min([bbox[0] for bbox in row])
            min_y = min([bbox[1] for bbox in row])
            max_x = max([bbox[0] + bbox[2] for bbox in row])
            max_y = max([bbox[1] + bbox[3] for bbox in row])
            rows_xyxy.append([min_x, min_y, max_x, max_y])
        
        cols_xyxy = []
        for key1, value1 in columns.items():
            if value1:
                min_x = min([bbox[0] for bbox in value1])
                min_y = min([bbox[1] for bbox in value1])
                max_x = max([bbox[0] + bbox[2] for bbox in value1])
                max_y = max([bbox[1] + bbox[3] for bbox in value1])
                cols_xyxy.append([min_x, min_y, max_x, max_y])

    with open(roi_txt_path, "r") as file:
        max_area = 0
        for line in file:
            elements = line.strip().split()
            if int(elements[0]) == 3:
                area = float(elements[3]) * float(elements[4])
                if area > max_area:
                    max_area = area
                    bbox_for_label_3 = list(map(float, elements[1:]))
                

    image_path = os.path.join(layout_images, filename)
    image = cv2.imread(image_path)
    h_image, w_image, _ = image.shape
    table_Xc_n, table_Yc_n, w_table_n, h_table_n = bbox_for_label_3
    l_table = max(0, int((table_Xc_n - w_table_n / 2) * w_image))
    t_table = max(0, int((table_Yc_n - h_table_n / 2) * h_image))
    w_table = int(w_table_n * w_image)
    h_table = int(h_table_n * h_image)
    cropped_table = image[
        int(t_table) : int(t_table + h_table), int(l_table) : int(l_table + w_table)
    ]

    cv2.rectangle(
            image, (l_table, t_table), (l_table+w_table, t_table+h_table), (0, 0, 255), 2
        )




    adjusted_row_boxes = []
    for box in rows_xyxy:
        x1_abs, y1_abs, x2_abs, y2_abs = (
            (box[0] * w_image),
            (box[1] * h_image),
            (box[2] * w_image),
            (box[3] * h_image),
        )
        adjusted_x1 = 0  # x1_abs - x_table
        adjusted_y1 = max(0, y1_abs - t_table)
        adjusted_x2 = w_table  # x2_abs - x_table
        adjusted_y2 = y2_abs - t_table
        if min((adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)) >= 0:
            adjusted_row_boxes.append(
                (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)
            )
        else:
            pass

    adjusted_col_boxes = []
    for box in cols_xyxy:
        x1_abs, y1_abs, x2_abs, y2_abs = (
            (box[0] * w_image),
            (box[1] * h_image),
            (box[2] * w_image),
            (box[3] * h_image),
        )
        adjusted_x1 = max(0, x1_abs - l_table)
        adjusted_y1 = y1_abs - t_table
        adjusted_x2 = x2_abs - l_table
        adjusted_y2 =  y2_abs - t_table
        if min((adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)) >= 0:
            adjusted_col_boxes.append(
                (adjusted_x1, adjusted_y1, adjusted_x2, adjusted_y2)
            )
        else:
            pass
            

    if not adjusted_row_boxes and not adjusted_col_boxes:
        continue

    cropped_table_path = os.path.join(images_folder, filename)
    cv2.imwrite(cropped_table_path, cropped_table)

    with open(output_file_path, "w") as file:
        for bbox in adjusted_row_boxes:
            row_w = bbox[2] - bbox[0]
            row_h = bbox[3] - bbox[1]
            row_bbox = [
                (bbox[0] + row_w / 2) / w_table,
                (bbox[1] + row_h / 2) / h_table,
                row_w / w_table,
                row_h / h_table,
            ]

            x1, y1 = int(bbox[0] + l_table), int(bbox[1]+t_table)
            x2,y2 = int(x1+ row_w) , int(y1 + row_h)

            if max(row_bbox) > 1:
                continue

            cv2.rectangle(
            image, (x1,y1), (x2,y2), (255, 0, 0), 2
            ) 


            file.write(f"0 {' '.join(map(str, row_bbox))}\n")
        for bbox in adjusted_col_boxes:
            col_w = bbox[2] - bbox[0]
            col_h = bbox[3] - bbox[1]
            col_bbox = [
                (bbox[0] + col_w / 2) / w_table,
                (bbox[1] + col_h / 2) / h_table,
                col_w / w_table,
                col_h / h_table,
            ]


            if max(col_bbox) > 1:
                continue

            x1, y1 = int(bbox[0] + l_table), int(bbox[1]+t_table)
            x2,y2 = int(x1 + col_w) , int(y1 + col_h)


            cv2.rectangle(
            image, (x1,y1), (x2,y2), (0, 255, 0), 2
            ) 


            file.write(f"1 {' '.join(map(str, col_bbox))}\n")


    output_path = "rows_cols_image.jpg"
    cv2.imwrite(os.path.join(current_dir, output_path), image)
    pass

print(number_of_samples)