import torch, os
import cv2
import time

current_dir = os.path.dirname(__file__)

model_path = os.path.join(current_dir, "best.pt")


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

        for result in results:
            min_x, min_y, w, h, conf, label = result

            # Convert normalized coordinates to absolute pixel values
            x1 = int((min_x - w / 2) * width)
            y1 = int((min_y - h / 2) * height)
            x2 = int((min_x + w / 2) * width)
            y2 = int((min_y + h / 2) * height)

            # Draw the bounding box
            cv2.rectangle(cv2_image_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Optionally, add the confidence score and label
            label_text = f'{int(label)}: {conf:.2f}'
            cv2.putText(cv2_image_copy, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        output_path = os.path.join(current_dir, "roi_output.jpg")

        # Save the image with bounding boxes
        cv2.imwrite(output_path, cv2_image_copy)

        return results


if __name__ == "__main__":
    filename = os.path.join(current_dir, "sample.jpg")

    roi_extractor = yolov5_roi()
    start_time = time.time()
    image = cv2.imread(filename)
    out = roi_extractor.get_roi(image)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
    print(out)
