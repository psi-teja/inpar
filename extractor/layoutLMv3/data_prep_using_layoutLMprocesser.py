import json
import os
from PIL import Image
from utils import processor_with_ocr, tokenizer, calculate_ioa, calculate_iou, preprocess_image_cv, preprocess_image_gray
from PIL import Image, ImageDraw, ImageFont


from tqdm import tqdm

current_dir = os.path.dirname(__file__)
datasetFolder = os.path.join(current_dir,"datasets","all")

imagesFolder = os.path.join(datasetFolder, "images")
extractedDataFolder = os.path.join(datasetFolder, "extracted_data")
labelJsonFolder = os.path.join(datasetFolder, "label_json")

images_list = os.listdir(imagesFolder)

label_count = {}

show_image = False


for i in tqdm(range(len(images_list)), unit="sample"):

    imageFileName = images_list[i]
    imagePath = os.path.join(imagesFolder, imageFileName)

    doc_id = imageFileName.split(".")[0]

    jsonPath = os.path.join(extractedDataFolder, f"{doc_id}.json")

    with open(jsonPath, "r") as file:
        extractedDataJson = json.load(file)

    labelJson = {"id": doc_id}

    image = Image.open(imagePath)
    imgW = image.width
    imgH = image.height

    encoding = processor_with_ocr(image)

    input_ids = encoding['input_ids'][0]
    bbox =  encoding['bbox'][0]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    words_list = []
    bboxes = []
    ner_tags = []

    font_path = "arial.ttf"  # Path to your font file
    font_size = 12
    if show_image:
        draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    curr_box = None

    segment_tokens = []
    for box, token in zip(bbox,tokens):

        if box == [0,0,0,0]: continue

        if box != curr_box and segment_tokens:
            words = tokenizer.convert_tokens_to_string(segment_tokens)
            
            words = words.replace("_","")
            words = words.replace("|","")

            words = words.strip()

            words_list.append(words)
            bboxes.append(curr_box)

            x1,y1,x2,y2 = curr_box
            x1 = imgW*(x1/1000)
            y1 = imgH*(y1/1000)
            x2 = imgW*(x2/1000)
            y2 = imgH*(y2/1000)

            w1 = x2-x1
            h1 = y2-y1

            if show_image:
                draw.rectangle([x1,y1,x2,y2], outline="red")
                draw.text((x1, y1 - 10), words, fill="blue", font=font)

            label = "Other"

            max_iou = 0

            for fieldName in extractedDataJson:
                fieldValue = extractedDataJson[fieldName]
                if isinstance(fieldValue, dict):
                    if fieldValue["location"]["ltwh"] == [0, 0, 0, 0]:
                        continue


                    ltwh = fieldValue["location"]["ltwh"]

                    l2, t2, w2, h2 = (
                        ltwh[0] * imgW,
                        ltwh[1] * imgH,
                        ltwh[2] * imgW,
                        ltwh[3] * imgH,
                    )

                    if calculate_ioa(x1, y1, w1, h1, l2, t2, w2, h2) > 0.5 and words: 
                        iou = calculate_iou(x1, y1, w1, h1, l2, t2, w2, h2)
                        if iou > max_iou:
                            max_iou = iou
                            label = fieldName
                            label = label.replace(" ","")

                elif isinstance(fieldValue, list):
                    if not fieldValue: continue

                    lineItems = fieldValue

                    for lineItem in lineItems:
                        for columnName in lineItem:
                            columnValue = lineItem[columnName]
                            if columnValue["location"]["ltwh"] == [0, 0, 0, 0]:
                                continue

                            ltwh = columnValue["location"]["ltwh"]

                            l2, t2, w2, h2 = (
                                ltwh[0] * imgW,
                                ltwh[1] * imgH,
                                ltwh[2] * imgW,
                                ltwh[3] * imgH,
                            )

                            if calculate_ioa(x1, y1, w1, h1, l2, t2, w2, h2) > 0.9 and words: 
                                iou = calculate_iou(x1, y1, w1, h1, l2, t2, w2, h2)
                                if iou > max_iou:
                                    max_iou = iou
                                    label = fieldName+columnName
                                    label = label.replace(" ","")

        
            ner_tags.append(label)

            if label in label_count:
                label_count[label]+= 1
            else:
                label_count[label]=1

            if show_image:
                draw.text((x1, y2), label, fill="red", font=font)

            curr_box = box
            segment_tokens = []
            segment_tokens.append(token)
        else:
            if not curr_box: 
                curr_box = box
            segment_tokens.append(token)


    labelJson["tokens"] = words_list
    labelJson["ner_tags"] = ner_tags
    labelJson["bboxes"] = bboxes

    if show_image:
        image.show()

    with open(os.path.join(labelJsonFolder, f'{doc_id}.json'), 'w') as json_file:
        json.dump(labelJson, json_file, indent=4)



details_file = os.path.join(datasetFolder, "details.cfg")

with open(details_file, "w") as f:
    f.write("[General]\n")
    f.write(f"NumberOfSamples: {len(images_list)}\n\n")
    f.write("[LabelCounts]\n")
    for label in label_count:
        f.write(f"{label} = {label_count[label]}\n")