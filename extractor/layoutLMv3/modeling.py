import torch, json, os
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Config
from PIL import Image, ImageDraw, ImageFont
import torch.nn.functional as F
from layoutLMv3_utils import tokenizer, ProcessorWithAWSOCR
import time

# Set environment variable for AWS profile
os.environ['AWS_PROFILE'] = "Developers_custom1_tallydev-AI-381491826341"

current_dir = os.path.dirname(__file__)

job_dir = os.path.join(current_dir, "results", "runs", "20240618233420")

saved_model_dir = os.path.join(job_dir, "saved_model")

model_config = LayoutLMv3Config.from_pretrained(saved_model_dir)
model = LayoutLMv3ForTokenClassification(model_config)



class layoutLMv3:
    def __init__(self) -> None:
        self.model = LayoutLMv3ForTokenClassification(model_config)
        self.model.load_state_dict(torch.load(os.path.join(saved_model_dir, "model.pth"), map_location=torch.device('cpu')))
        self.model.eval()
        self.processer = ProcessorWithAWSOCR()

    def get_field_values(self, pil_image):

        # Preprocess image
        encodings = self.processer.get_encodings(pil_image)

        tokens = tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])

        draw = ImageDraw.Draw(pil_image)

        # Perform token classification
        with torch.no_grad():
            outputs = self.model(**encodings)

        bboxes = encodings["bbox"][0].tolist()

        imgW = pil_image.width
        imgH = pil_image.height

        unique_bboxes_xyxy= []
        words_in_unique_bboxes = []
        labels_for_unique_bboxes = []
        conf_arr_of_unique_bboxes = []

        curr_box = None
        segment_tokens = []
        conf_arr = []

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

                x1, y1, x2, y2 = curr_box

                x1 = (x1 / 1000)
                y1 = (y1 / 1000)
                x2 = (x2 / 1000)
                y2 = (y2 / 1000)

                unique_bboxes_xyxy.append([x1,y1,x2,y2])
                words_in_unique_bboxes.append(words)

                labels_for_unique_bboxes.append(self.model.config.id2label[label_id])
                conf_arr_of_unique_bboxes.append(conf_arr)


                curr_box = box
                label_index = int(softmax_output.argmax(-1))
                conf = softmax_output[label_index]
                conf_arr = [(self.model.config.id2label[label_index], conf.item())]
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
                conf_arr.append((self.model.config.id2label[label_index], conf.item()))
                if conf > max_conf:
                    max_conf = conf
                    label_id = label_index

        return unique_bboxes_xyxy, words_in_unique_bboxes, labels_for_unique_bboxes, conf_arr_of_unique_bboxes

        


if __name__ == "__main__":

    image_path = os.path.join(current_dir, "sample.jpg")

    layoutLMv3_extractor = layoutLMv3()
    start_time = time.time()
    pil_image = Image.open(image_path)
    field_values = layoutLMv3_extractor.get_field_values(pil_image)
    end_time = time.time()
    print("Time to run inference code:", end_time - start_time)
    with open(os.path.join(current_dir, "sample_output.json"), "w") as json_file:
        json.dump(field_values, json_file)
    pil_image.save(os.path.join(current_dir, "sample_output.jpg"))
