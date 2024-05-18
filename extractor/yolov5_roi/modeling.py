import torch, os

current_dir = os.path.dirname(__file__)

msiz = 500

model_path = os.path.join(current_dir, "best.pt")

class yolov5_roi():
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=True)
    
    def get_roi(self, image):
        pred = self.model(image, size = msiz)
        pred = pred.xywhn[0]
        result = pred.cpu().numpy()
        return result