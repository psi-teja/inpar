import torch, os
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from layoutLMv3_utils import (
    tokenizer,
    ProcessorWithEASYOCR,
    get_label_from_sum_conf,
    biased_softmax,
    calculate_ioa,
    remove_unnecessary_tokens,
)
import time

current_dir = os.path.dirname(__file__)

job_id = "20240705105520"
job_dir = os.path.join(current_dir, "results", "runs", job_id)

saved_model_dir = os.path.join(job_dir, "saved_model")

model_config = LayoutLMv3Config.from_pretrained(saved_model_dir)
model = LayoutLMv3ForTokenClassification(model_config)


class layoutLMv3:
    def __init__(self) -> None:
        self.model = LayoutLMv3ForTokenClassification(model_config)
        self.model.load_state_dict(
            torch.load(
                os.path.join(saved_model_dir, "model.pth"),
                map_location=torch.device("cpu"),
            )
        )
        self.model.eval()
        self.processer = ProcessorWithEASYOCR()
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id

    def get_field_values(self, pil_image, roi_data):

        # Preprocess image
        encodings = self.processer.get_encodings(pil_image)

        all_tokens = []
        all_bboxes = []
        all_output_logits = []


        for i in range(0, len(encodings["input_ids"][0]), 512):

            chuck_encodings = {
                "input_ids" : torch.tensor([encodings["input_ids"][0][i:i+512].tolist()]),
                "attention_mask": torch.tensor([encodings['attention_mask'][0][i:i+512].tolist()]),
                "bbox" : torch.tensor([encodings['bbox'][0][i:i+512].tolist()]),
                "pixel_values" : encodings['pixel_values']
            }

            tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0][i:i+512])
            bboxes = encodings["bbox"][0][i:i+512].tolist()
            # Perform token classification
            with torch.no_grad():
                outputs = self.model(**chuck_encodings)

            output_logits = outputs.logits[0]

            all_tokens.extend(tokens)
            all_bboxes.extend(bboxes)
            all_output_logits.extend(output_logits)

        draw = ImageDraw.Draw(pil_image)
        font = ImageFont.load_default(size=20)

        imgW = pil_image.width
        imgH = pil_image.height

        unique_bboxes_xyxy = []
        words_in_unique_bboxes = []
        labels_for_unique_bboxes = []
        conf_arr_of_unique_bboxes = []

        curr_box = None
        segment_tokens = []
        conf_arr = []

        max_conf = 0

        for box, token, logits in zip(all_bboxes, all_tokens, all_output_logits):

            if box == [0, 0, 0, 0]:
                continue

            if not roi_data:
                softmax_output = F.softmax(logits, dim=0)
            else:
                best_roi_class = None
                max_conf = 0
                for roi_result in roi_data:
                    Xc, Yc, w1, h1, conf, class_name = roi_result
                    l2, t2, w2, h2 = [
                        (box[0] / 1000),
                        (box[1] / 1000),
                        ((box[2] - box[0]) / 1000),
                        ((box[3] - box[1]) / 1000),
                    ]
                    ioa = calculate_ioa(Xc-w1/2, Yc-h1/2, w1, h1, l2, t2, w2, h2)

                    cus_conf = ioa * conf

                    if ioa > 0.5 and cus_conf > max_conf:
                        max_conf = cus_conf
                        best_roi_class = class_name

                softmax_output = biased_softmax(logits, best_roi_class, self.label2id, bias=10)

            if box != curr_box and segment_tokens:
               

                start_index = remove_unnecessary_tokens(segment_tokens)

                words = tokenizer.convert_tokens_to_string(segment_tokens[start_index:])

                words = words.strip()

                if words == "":
                    continue

                x1, y1, x2, y2 = curr_box

                x1 = x1 / 1000
                y1 = y1 / 1000
                x2 = x2 / 1000
                y2 = y2 / 1000

                unique_bboxes_xyxy.append([x1, y1, x2, y2])
                words_in_unique_bboxes.append(words)

                label_text = get_label_from_sum_conf(conf_arr[start_index:])
                # label_text = self.model.config.id2label[label_id]

                labels_for_unique_bboxes.append(label_text)
                conf_arr_of_unique_bboxes.append(conf_arr[start_index:])

                x1 *= imgW
                y1 *= imgH
                x2 *= imgW
                y2 *= imgH

                if label_text != "Other":
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1, y1 - 30), f"{label_text}", fill="red", font=font)
                output_path = os.path.join(current_dir, "token_classication_output.jpg")
                pil_image.save(output_path)

                curr_box = box
                curr_label_index = int(softmax_output.argmax(-1))
                curr_conf = softmax_output[curr_label_index]
                curr_label_text = self.model.config.id2label[curr_label_index]
                conf_arr = [(curr_label_text, curr_conf.item())]
                max_conf = conf
                segment_tokens = [token]
            else:
                if not curr_box:
                    curr_box = box
                segment_tokens.append(token)
                label_index = int(softmax_output.argmax(-1))
                conf = softmax_output[label_index]
                conf_arr.append((self.model.config.id2label[label_index], conf.item()))
                if conf > max_conf:
                    max_conf = conf
                    label_id = label_index

        return (
            unique_bboxes_xyxy,
            words_in_unique_bboxes,
            labels_for_unique_bboxes,
            conf_arr_of_unique_bboxes,
        )

def split_into_chunks(data, chunk_size=512):
    """Splits data into chunks of size chunk_size."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


if __name__ == "__main__":

    # image_path = os.path.join(current_dir, "sample.jpg")

    image_path = "/home/saiteja/DocAI/code.tallyai/AIBackend/DocAI/inpar-research/extractor/layoutLMv3/datasets/imerit/phase1/images/171534753673582784290-7700-4f85-b563-6b5ee41ecae7_page1.jpeg"

    layoutLMv3_extractor = layoutLMv3()
    start_time = time.time()
    pil_image = Image.open(image_path)
    field_values = layoutLMv3_extractor.get_field_values(pil_image, roi_data=[])
    end_time = time.time()
    print("Time(sec) to run inference code:", end_time - start_time)
