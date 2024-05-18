import torch, json, os
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from layoutLMv3_utils import tokenizer, processor_with_ocr

current_dir = os.path.dirname(__file__)

job_id = "20240420232435"
job_dir = os.path.join(current_dir, "results", "runs", job_id)

saved_model_dir = os.path.join(job_dir, "saved_model")

model_config = LayoutLMv3Config.from_pretrained(saved_model_dir)
model = LayoutLMv3ForTokenClassification(model_config)


class layoutLMv3():
    def __init__(self) -> None:
        self.model = LayoutLMv3ForTokenClassification(model_config)
        self.model.load_state_dict(torch.load(os.path.join(saved_model_dir, "model.pth"), map_location=torch.device('cpu')))
        self.processer = processor_with_ocr

    def get_field_values(self, image):

        # Preprocess image
        inputs = processor_with_ocr(
            image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        draw = ImageDraw.Draw(image)

        # Perform token classification
        with torch.no_grad():
            outputs = self.model(**inputs)

        bboxes = inputs["bbox"][0].tolist()

        imgW = image.width
        imgH = image.height

        unique_bboxes = []
        words_in_unique_bboxes = []
        labels_for_unique_bboxes = []

        curr_box = None
        segment_tokens = []

        max_conf = 0

        for box, token, logits in zip(bboxes, tokens, outputs.logits[0]):

            if box == [0, 0, 0, 0]:
                continue

            softmax_output = F.softmax(logits, dim=0)

            if box != curr_box and segment_tokens:
                words = tokenizer.convert_tokens_to_string(segment_tokens)

                words = words.replace("_", "")
                words = words.replace("|", "")

                words = words.strip()

                unique_bboxes.append(curr_box)
                words_in_unique_bboxes.append(words)

                labels_for_unique_bboxes.append(self.model.config.id2label[label_id])
                curr_box = box
                label_index = int(softmax_output.argmax(-1))
                conf = softmax_output[label_index]
                max_conf = conf
                label_id = label_index
                segment_tokens = []
                segment_tokens.append(token)
            else:
                if not curr_box:
                    curr_box = box
                segment_tokens.append(token)
                label_index = int(softmax_output.argmax(-1))
                conf = softmax_output[label_index]
                if conf > max_conf:
                    max_conf = conf
                    label_id = label_index

        field_values = {}

        for bbox, words, label in zip(
            unique_bboxes, words_in_unique_bboxes, labels_for_unique_bboxes
        ):

            x1, y1, x2, y2 = bbox

            x1 = imgW * (x1 / 1000)
            y1 = imgH * (y1 / 1000)
            x2 = imgW * (x2 / 1000)
            y2 = imgH * (y2 / 1000)

            if label == "Other":
                continue

            box_height = y2 - y1

            font_size = max(
                int(box_height * 0.2), 12
            )  

            font = ImageFont.truetype("arial.ttf", size=font_size)

            draw.rectangle([x1, y1, x2, y2], outline="red")

            draw.text((x1, y1 - font_size), label, fill="blue", font=font)


            if label in field_values:
                field_values[label].append({"text": words,"location": {"pageNo": 1,"ltwh": [x1/imgW,y1/imgH,(x2-x1)/imgW,(y2-y1)/imgH]}})

            else:
                field_values[label] = [{"text": words,"location": {"pageNo": 1,"ltwh": [x1/imgW,y1/imgH,(x2-x1)/imgW,(y2-y1)/imgH]}}]


        return field_values

if __name__ == "__main__":

    image_path = os.path.join(current_dir, "sample.jpg")

    image = Image.open(image_path)

    layoutLMv3_extractor = layoutLMv3()

    field_values = layoutLMv3_extractor.get_field_values(image)

    with open(os.path.join(current_dir, "sample_output.json"), 'w') as json_file:
        json.dump(field_values, json_file)

    image.save(os.path.join(current_dir, "sample_output.jpg"))
