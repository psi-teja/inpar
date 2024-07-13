import torch
import os
import cv2
import time
from yolov5_roi_utils import download_model_from_s3

current_dir = os.path.dirname(__file__)
job_dir = os.path.join(current_dir, "results","runs","20240618233421")

if not os.path.exists(job_dir):
    if not download_model_from_s3(job_dir):
        raise RuntimeError("Failed to download model from S3. Exiting script.")

model_path = os.path.join(job_dir, "weights", "best.pt")

class yolov5_roi:
    def __init__(self) -> None:
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", model_path, force_reload=True
        )
        self.msiz = 500

    def get_roi(self, cv2_image):
        pred = self.model(cv2_image, size=self.msiz)
        pred = pred.xywhn[0]
        results = pred.cpu().numpy()

        height, width, _ = cv2_image.shape
        cv2_image_copy = cv2_image.copy()

        updated_results = []

        for result in results:
            min_x, min_y, w, h, conf, class_id = result

            # Convert normalized coordinates to absolute pixel values
            x1 = int((min_x - w / 2) * width)
            y1 = int((min_y - h / 2) * height)
            x2 = int((min_x + w / 2) * width)
            y2 = int((min_y + h / 2) * height)

            

            # Get class name from class_id
            class_name = self.model.names[int(class_id)]

            # Replace class_id with class_name in the result
            updated_results.append([min_x, min_y, w, h, conf, class_name])

            # Optionally, add the confidence score and label
            label_text = f'{class_name}: {conf:.2f}'

            # Draw the bounding box
            if float(conf) > 0.8:
                cv2.rectangle(cv2_image_copy, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
                cv2.putText(cv2_image_copy, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)

        output_path = os.path.join(current_dir, "roi_output.jpg")

        # Save the image with bounding boxes
        cv2.imwrite(output_path, cv2_image_copy)

        return updated_results

if __name__ == "__main__":
    filename = os.path.join(current_dir, "sample.jpg")

    roi_extractor = yolov5_roi()
    start_time = time.time()
    image = cv2.imread(filename)
    out = roi_extractor.get_roi(image)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
    print(out)
