import torch, os
import cv2
import asyncio

current_dir = os.path.dirname(__file__)
msiz = 500
model_path = os.path.join(current_dir, "best.pt")

# import pathlib
# from pathlib import Path

# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath


class yolov5_roi:
    def __init__(self, cv2_queue, roi_queue) -> None:
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", model_path, force_reload=True
        )
        self.roi_queue = roi_queue
        self.cv2_queue = cv2_queue

    async def get_roi(self):
        if not self.cv2_queue.empty():
            image = await self.cv2_queue.get()
            pred = self.model(image, size=msiz)
            pred = pred.xywhn[0]
            result = pred.cpu().numpy()
            await self.roi_queue.put(result)
        else:
            await asyncio.sleep(1)


# async def main():
#     filename = os.path.join(current_dir, "sample.jpg")
#     image = cv2.imread(filename)
#     roi_queue = asyncio.Queue()

#     yolov5_instance = yolov5_roi(roi_queue)
#     await yolov5_instance.get_roi(image)

#     roi_result = None
#     if not roi_queue.empty():
#         roi_result = await roi_queue.get()
#     else:
#         await asyncio.sleep(1)

#     print(roi_result)


# if __name__ == "__main__":
#     asyncio.run(main())
