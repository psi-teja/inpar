import torch
import os
import cv2

current_dir = os.path.dirname(__file__)

msiz = 500

model_path = os.path.join(current_dir, "yolov5_table.pt")

class yolov5_table():
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, force_reload=True)
    def get_table_structure(self, cv2_image, table_ltwh):
        l,t,w,h = table_ltwh
        height, width, _ = cv2_image.shape

        x1 = max(0, int((l-w/2-0.001)*width))
        y1 = max(0, int((t-h/2-0.001)*height))
        x2 = min(width, int((l+w/2+0.001)*width))
        y2 = min(height, int((t+h/2+0.001)*height))

        table_cv2_image = cv2_image[y1:y2, x1:x2, :]
        
        pred = self.model(table_cv2_image, size = msiz)
        pred = pred.xywhn[0]
        results = pred.cpu().numpy()
        
        output = {"column_boxes":[],
                  "row_boxes":[]}

        w_t = int(x2-x1)
        h_t = int(y2-y1)
        xoff, yoff = x1, y1

        for result in results:
            min_x = result[0]
            min_y = result[1]
            w = result[2]
            h = result[3]
            
            x1 = ((min_x-w/2)*w_t + xoff)/width
            y1 = ((min_y-h/2)*h_t + yoff)/height
            x2 = ((min_x+w/2)*w_t + xoff)/width
            y2 = ((min_y+h/2)*h_t + yoff)/height

            label = result[-1]

            conf = result[-2]

            if label == 1:
                output["column_boxes"].append({'bbox':[x1, y1, x2, y2],'score':conf})
            if label == 2:
                output["row_boxes"].append({'bbox':[x1, y1, x2, y2],'score':conf})

        return output

    
if __name__ == "__main__":
    filename = os.path.join(current_dir, '20240417092531955.jpg')
    image = cv2.imread(filename)
    out = yolov5_table().get_table_structure(image)
    print(out)