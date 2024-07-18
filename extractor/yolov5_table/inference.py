import torch
import os
import cv2
import numpy as np
from PIL import Image
from yolov5_table_utils import get_random_color

current_dir = os.path.dirname(__file__)
job_dir = os.path.join(current_dir, "results","runs","20240618233422")

model_path = os.path.join(job_dir, "weights", "best.pt")

class yolov5_table:
    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", model_path, force_reload=True
        )

        self.msiz = 500

    def get_table_structure(
        self,
        cv2_image,
        table_XcYcWH,
        tally_ai_json,
        unique_bboxes_xyxy,
        words_in_unique_bboxes,
        labels_for_unique_bboxes,
        conf_arr_of_unique_bboxes,
        pageNo,
    ):

        if not list(table_XcYcWH):
            return []
        Xc, Yc, W, H = table_XcYcWH
        height, width, _ = cv2_image.shape
        cv2_image_copy = cv2_image.copy()
        x1 = max(0, int((Xc - W / 2 - 0.001) * width))
        y1 = max(0, int((Yc - H / 2 - 0.001) * height))
        x2 = min(width, int((Xc + W / 2 + 0.001) * width))
        y2 = min(height, int((Yc + H / 2 + 0.001) * height))
        table_cv2_image = cv2_image[y1:y2, x1:x2, :]

        M, warped_image = perspective_transform(table_cv2_image)
        warped_image_copy = warped_image.copy()
        warped_image_copy2 = warped_image.copy()
        pred = self.model(warped_image, size=self.msiz)
        pred = pred.xywhn[0]
        results = pred.cpu().numpy()

        output = {"column_boxes": [], "row_boxes": []}

        h_t, w_t, _ = warped_image.shape
        xoff, yoff = x1 / width, y1 / height

        for result in results:
            min_x = result[0]
            min_y = result[1]
            W = result[2]
            H = result[3]

            new_x11 = (min_x - W / 2) * w_t
            new_y11 = (min_y - H / 2) * h_t
            new_x21 = (min_x + W / 2) * w_t
            new_y21 = (min_y + H / 2) * h_t
            label = result[-1]
            conf = result[-2]

            new_x1 = ((min_x - W / 2) * w_t) / width
            new_y1 = ((min_y - H / 2) * h_t) / height
            new_x2 = ((min_x + W / 2) * w_t) / width
            new_y2 = ((min_y + H / 2) * h_t) / height

            if label == 1:
                if conf > 0.5:
                    box_color = get_random_color()
                    output["column_boxes"].append(
                        {"bbox": [new_x1, new_y1, new_x2, new_y2], "score": conf}
                    )
            elif label == 0:
                box_color = get_random_color()
                if conf > 0.5:
                    output["row_boxes"].append(
                        {"bbox": [new_x1, new_y1, new_x2, new_y2], "score": conf}
                    )

            if conf > 0.5:
                cv2.rectangle(
                    warped_image_copy,
                    (int(new_x11), int(new_y11)),
                    (int(new_x21), int(new_y21)),
                    box_color,
                    4,
                )
                output_image_path = os.path.join(
                    current_dir, "wrapped_row_col_output.jpg"
                )
                cv2.imwrite(output_image_path, warped_image_copy)

        table_row_wrt_page = output["row_boxes"]

        table_row_wrt_page = sorted(
            table_row_wrt_page, key=lambda item: item["bbox"][1]
        )

        num_of_rows = len(table_row_wrt_page)

        row_template = tally_ai_json["Table"][0]

        tally_ai_json["Table"] = [dict(row_template) for _ in range(num_of_rows)]

        for bbox, words, label_layout, conf_arr in zip(
            unique_bboxes_xyxy,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
        ):

            if label_layout[:5] != "Table":
                continue

            orig_x1, orig_y1, orig_x2, orig_y2 = bbox
            orig_l1 = orig_x1
            orig_t1 = orig_y1
            orig_w1 = orig_x2 - orig_x1
            orig_h1 = orig_y2 - orig_y1
            orig_ltwh1 = [orig_l1, orig_t1, orig_w1, orig_h1]

            table_x1 = int((orig_x1 - xoff) * width)
            table_y1 = int((orig_y1 - yoff) * height)
            table_x2 = int((orig_x2 - xoff) * width)
            table_y2 = int((orig_y2 - yoff) * height)
            if table_x1 < 0 or table_y1 < 0 or table_x2 < 0 or table_y2 < 0:
                continue

            transformed_table_bbox = np.array(
                [
                    [table_x1, table_y1],
                    [table_x2, table_y1],
                    [table_x2, table_y2],
                    [table_x1, table_y2],
                ],
                dtype=np.float32,
            )
            transformed_bbox = cv2.perspectiveTransform(
                np.array([transformed_table_bbox]), M
            )[0]
            new_x1, new_y1 = transformed_bbox[0]
            new_x2, new_y2 = transformed_bbox[2]

            cv2.rectangle(
                warped_image_copy2,
                (int(new_x1), int(new_y1)),
                (int(new_x2), int(new_y2)),
                (0, 250, 0),
                4,
            )
            output_image_path = os.path.join(
                current_dir, "wrapped_row_col_text_output.jpg"
            )
            cv2.imwrite(output_image_path, warped_image_copy2)

            l1 = new_x1 / width
            t1 = new_y1 / height
            w1 = (new_x2 - new_x1) / width
            h1 = (new_y2 - new_y1) / height
            ltwh1 = [l1, t1, w1, h1]

            if label_layout.find("Table") != -1:
                if not table_row_wrt_page:
                    continue
                row_index = find_items_table_row(table_row_wrt_page, ltwh1)
                if row_index == None:
                    continue
                col_name = label_layout[5:]
                extractedValue = tally_ai_json["Table"][row_index][col_name]

                if extractedValue["location"]["pageNo"] == 0:
                    tally_ai_json["Table"][row_index][col_name] = {
                        "text": words,
                        "location": {
                            "pageNo": pageNo,
                            "ltwh": orig_ltwh1,
                        },
                    }
                else:
                    new_text = extractedValue["text"] + " " + words
                    old_ltwh = extractedValue["location"]["ltwh"]
                    new_ltwh = min_super_rectangle(old_ltwh, orig_ltwh1)
                    tally_ai_json["Table"][row_index][col_name] = {
                        "text": new_text,
                        "location": {
                            "pageNo": pageNo,
                            "ltwh": new_ltwh,
                        },
                    }

        return tally_ai_json


def perspective_transform(table_cv2_image):
    output_image_path = os.path.join(current_dir, "cropped_iamge.jpg")
    cv2.imwrite(output_image_path, table_cv2_image)

    gray_image = cv2.cvtColor(table_cv2_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.4)
    # thresholded_image = cv2.adaptiveThreshold(
    #     blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
    # )
    edges = cv2.Canny(blurred_image, 10, 280, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    output_image_path = os.path.join(current_dir, "contours_iamge.jpg")
    dummy_image = table_cv2_image.copy()
    cv2.drawContours(dummy_image, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(output_image_path, dummy_image)

    extreme_points_contour = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        # Find the extreme points
        top_left = contour[np.argmin(np.sum(contour, axis=1))]
        top_right = contour[np.argmin(np.diff(contour, axis=1))]
        bottom_left = contour[np.argmax(np.diff(contour, axis=1))]
        bottom_right = contour[np.argmax(np.sum(contour, axis=1))]

        extreme_points = np.array(
            [top_left, bottom_left, bottom_right, top_right], dtype=np.int32
        )

        # Reshape to contour format (N, 1, 2)
        extreme_points_contour.append(extreme_points)

    all_points = np.vstack(extreme_points_contour)

    top_left = all_points[np.argmin(np.sum(all_points, axis=1))]
    top_right = all_points[np.argmin(np.diff(all_points, axis=1))]
    bottom_left = all_points[np.argmax(np.diff(all_points, axis=1))]
    bottom_right = all_points[np.argmax(np.sum(all_points, axis=1))]

    # min_x = np.min(all_points[:, 0])
    # max_x = np.max(all_points[:, 0])
    # min_y = np.min(all_points[:, 1])
    # max_y = np.max(all_points[:, 1])
    # top_left = (min_x, min_y)
    # bottom_right = (max_x, max_y)
    # top_right = (max_x, min_y)
    # bottom_left = (min_x, max_y)

    extreme_points = np.array(
        [list(top_left), list(top_right), list(bottom_right), list(bottom_left)],
        dtype=np.int32,
    )

    imgh, imgw, _ = table_cv2_image.shape

    # thresh = 5
    # if top_left[0] - thresh >= 0:
    #     top_left[0] = top_left[0] - thresh
    # else:
    #     top_left[0] = top_left[0]
    # if top_left[1] - thresh >= 0:
    #     top_left[1] = top_left[1] - thresh
    # else:
    #     top_left[1] = top_left[1]
    # if top_right[0] + thresh <= imgw:
    #     top_right[0] = top_right[0] + thresh
    # else:
    #     top_right[0] = top_right[0]
    # if top_right[1] - thresh >= 0:
    #     top_right[1] = top_right[1] - thresh
    # else:
    #     top_right[1] = top_right[1]
    # if bottom_left[0] + thresh <= imgw:
    #     bottom_left[0] = bottom_left[0] + thresh
    # else:
    #     bottom_left[0] = bottom_left[0]
    # if bottom_left[1] + thresh <= imgh:
    #     bottom_left[1] = bottom_left[1] + thresh
    # else:
    #     bottom_left[1] = bottom_left[1]
    # if bottom_left[0] - thresh >= 0:
    #     bottom_left[0] = bottom_left[0] - thresh
    # else:
    #     bottom_left[0] = bottom_left[0]
    # if bottom_left[1] + thresh <= imgh:
    #     bottom_left[1] = bottom_left[1] + thresh
    # else:
    #     bottom_left[1] = bottom_left[1]

    right_h = bottom_right[1] - top_right[1]
    left_h = bottom_left[1] - top_left[1]
    if right_h > left_h:
        if (top_left[1] - abs(right_h - left_h)) >= 0:
            top_left[1] = top_left[1] - abs(right_h - left_h)
        else:
            bottom_left[1] = bottom_left[1] + abs(right_h - left_h) - top_left[1]
            top_left[1] = 0
    elif right_h < left_h:
        if (top_right[1] - abs(right_h - left_h)) >= 0:
            top_right[1] = top_right[1] - abs(right_h - left_h)
        else:
            bottom_right[1] = bottom_right[1] + abs(right_h - left_h) - top_right[1]
            top_right[1] = 0

    extreme_points_pad = np.array(
        [list(top_left), list(top_right), list(bottom_right), list(bottom_left)],
        dtype=np.int32,
    )

    # padding = np.argmax(
    #     [top_left[1], top_right[1], imgh - bottom_right[1], imgh - bottom_left[1]]
    # )
    # if padding == 0:
    #     top_left[1] = top_left[1] - abs(right_h - left_h)
    # elif padding == 1:
    #     top_right[1] = top_right[1] - abs(right_h - left_h)
    # elif padding == 2:
    #     bottom_right[1] = bottom_right[1] + abs(right_h - left_h)
    # else:
    #     bottom_left[1] = bottom_left[1] + abs(right_h - left_h)

    debug_image1 = table_cv2_image.copy()
    output_image_path = os.path.join(current_dir, "edges_iamge.jpg")
    cv2.drawContours(debug_image1, [extreme_points_pad], -1, (0, 255, 0), 2)
    cv2.imwrite(output_image_path, debug_image1)

    st_points = np.array(
        [list(top_left), list(top_right), list(bottom_right), list(bottom_left)],
        dtype=np.float32,
    )
    imgw_pad = int((imgw / imgh) * 25)
    imgh_pad = int((imgw / imgh) * 25) + 100  # int(imgw_pad * (imgw / imgh))
    dst_points = np.array(
        [
            [0, 0],
            [imgw + imgw_pad, 0],
            [imgw + imgw_pad, imgh + imgh_pad],
            [0, imgh + imgh_pad],
        ],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(st_points, dst_points)
    warped_image = cv2.warpPerspective(
        table_cv2_image, M, (imgw + imgw_pad, imgh + imgh_pad)
    )
    output_image_path = os.path.join(current_dir, "wrapped_iamge.jpg")
    cv2.imwrite(output_image_path, warped_image)
    return M, warped_image


def find_items_table_row(table_row_wrt_page, ltwh1):

    l1, t1, w1, h1 = ltwh1

    max_ioa = 0
    best_row = None

    for i in range(len(table_row_wrt_page)):
        x1, y1, x2, y2 = table_row_wrt_page[i]["bbox"]

        l2 = x1
        t2 = y1
        w2 = x2 - x1
        h2 = y2 - y1

        ioa = calculate_ioa(l1, t1, w1, h1, l2, t2, w2, h2)

        if max_ioa < ioa:
            max_ioa = ioa
            best_row = i

    return best_row


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


def min_super_rectangle(ltwh1, ltwh2):
    l1, t1, w1, h1 = ltwh1
    l2, t2, w2, h2 = ltwh2

    l = min(l1, l2)
    t = min(t1, t2)
    w = max(l1 + w1, l2 + w2) - l
    h = max(t1 + h1, t2 + h2) - t

    return [l, t, w, h]


if __name__ == "__main__":
    filename = os.path.join(current_dir, "sample.jpg")
    image = cv2.imread(filename)
    table_detector = yolov5_table()
    out = table_detector.get_table_structure(
        image, (0.5, 0.5, 1, 1)
    )  # Example table coordinates
    print(out)
