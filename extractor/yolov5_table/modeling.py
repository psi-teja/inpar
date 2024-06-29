import torch
import os
import cv2
import numpy as np
from PIL import Image

current_dir = os.path.dirname(__file__)
# model_path = os.path.join(current_dir, "yolov5_table.pt")
model_path = os.path.join(current_dir, "results", "runs", "exp6", "weights", "best.pt")

class yolov5_table:
    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", model_path, force_reload=True
        )

        self.msiz = 500

    def get_table_structure(self, cv2_image, table_XcYcWH):
        if not list(table_XcYcWH): return []
        Xc, Yc, W, H = table_XcYcWH
        height, width, _ = cv2_image.shape

        cv2_image_copy = cv2_image.copy()

        x1 = max(0, int((Xc - W / 2 - 0.001) * width))
        y1 = max(0, int((Yc - H / 2 - 0.001) * height))
        x2 = min(width, int((Xc + W / 2 + 0.001) * width))
        y2 = min(height, int((Yc + H / 2 + 0.001) * height))

        table_cv2_image = cv2_image[y1:y2, x1:x2, :]

        # rotated_image, angle = self.correct_tilt(table_cv2_image)

        pred = self.model(table_cv2_image, size=self.msiz)

        # pred = self.rotate_predictions(pred, angle, rotated_image.shape)

        pred = pred.xywhn[0]
        results = pred.cpu().numpy()

        output = {"column_boxes": [], "row_boxes": []}

        w_t = int(x2 - x1)
        h_t = int(y2 - y1)
        xoff, yoff = x1, y1

        for result in results:
            min_x = result[0]
            min_y = result[1]
            W = result[2]
            H = result[3]

            x1 = ((min_x - W / 2) * w_t + xoff) / width
            y1 = ((min_y - H / 2) * h_t + yoff) / height
            x2 = ((min_x + W / 2) * w_t + xoff) / width
            y2 = ((min_y + H / 2) * h_t + yoff) / height

            label = result[-1]
            conf = result[-2]

            if label == 1:
                output["column_boxes"].append({"bbox": [x1, y1, x2, y2], "score": conf})
            elif label == 0:
                output["row_boxes"].append({"bbox": [x1, y1, x2, y2], "score": conf})

            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height

            if conf>0.5:
                cv2.rectangle(cv2_image_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                output_image_path = os.path.join(current_dir, "row_col_output.jpg")
                cv2.imwrite(output_image_path, cv2_image_copy)

        return output["row_boxes"]

    def correct_tilt(self, table_cv2_image):
        gray_array = cv2.cvtColor(table_cv2_image, color=cv2.COLOR_BGR2GRAY)

        # Detect edges using the Canny edge detector
        edges = cv2.Canny(gray_array, 50, 150)

        # Detect lines using the Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return table_cv2_image, 0  # No lines detected, return original image

        # Calculate the angle of the lines
        angles = [np.degrees(theta) for rho, theta in lines[:, 0]]

        # Compute the median angle
        median_angle = np.median(angles) - 90  # Adjusting for the tilt

        # Rotate the image by the median angle
        image_pil = Image.fromarray(cv2.cvtColor(table_cv2_image, cv2.COLOR_BGR2RGB))
        rotated_image = image_pil.rotate(median_angle, resample=Image.BICUBIC, expand=True)

        # Convert the rotated image to OpenCV format for further processing
        rotated_array = np.array(rotated_image.convert("RGB"))
        rotated_gray = cv2.cvtColor(rotated_array, cv2.COLOR_RGB2GRAY)

        return rotated_gray, median_angle

    def rotate_predictions(self, pred, angle, image_shape):
        angle_rad = np.deg2rad(angle)
        center_x, center_y = image_shape[1] / 2, image_shape[0] / 2

        for result in pred.xywhn[0]:
            x, y, w, h = result[:4]

            x -= center_x
            y -= center_y

            new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
            new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)

            result[0] = new_x + center_x
            result[1] = new_y + center_y

        return pred


if __name__ == "__main__":
    filename = os.path.join(current_dir, "sample.jpg")
    image = cv2.imread(filename)
    table_detector = yolov5_table()
    out = table_detector.get_table_structure(image, (0.5, 0.5, 1, 1))  # Example table coordinates
    print(out)
