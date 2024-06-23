import torch, os
import cv2
import time

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.Path

current_dir = os.path.dirname(__file__)

msiz = 500

model_path = os.path.join(current_dir, "best.pt")


class yolov5_roi:
    def __init__(self) -> None:
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", model_path, force_reload=True
        )

    def get_roi(self, image):
        pred = self.model(image, size=msiz)
        pred = pred.xywhn[0]
        result = pred.cpu().numpy()
        return result


if __name__ == "__main__":
    filename = os.path.join(current_dir, "sample.jpg")

    roi_extractor = yolov5_roi()
    start_time = time.time()
    image = cv2.imread(filename)
    out = roi_extractor.get_roi(image)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
    print(out)
