import json
import os
from PIL import Image
import pytesseract
from utils import *
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
datasetFolder = os.path.join(current_dir,"datasets","tally")

imagesFolder = os.path.join(datasetFolder, "images")
extractedDataFolder = os.path.join(datasetFolder, "extracted_data")
labelJsonFolder = os.path.join(datasetFolder, "label_json")

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

images_list = os.listdir(imagesFolder)

label_count = {}

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

    ocr_data = pytesseract.image_to_data(image, output_type="dict")

    ocr_data["label"] = []

    tokens = []
    bboxes = []
    ner_tags = []

    for i in range(len(ocr_data["text"])):
        text = ocr_data["text"][i]
        l1, t1, w1, h1 = (
            ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i],
        )
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

                if calculate_ioa(l1, t1, w1, h1, l2, t2, w2, h2) > 0.98 and text: 
                    iou = calculate_iou(l1, t1, w1, h1, l2, t2, w2, h2)
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

                        if calculate_ioa(l1, t1, w1, h1, l2, t2, w2, h2) > 0.98 and text: 
                            iou = calculate_iou(l1, t1, w1, h1, l2, t2, w2, h2)
                            if iou > max_iou:
                                max_iou = iou
                                label = fieldName+columnName
                                label = label.replace(" ","")

        
        ocr_data["label"].append(label)

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        if not text: continue

        tokens.append(text)

        l,t,w,h = [ocr_data["left"][i],
            ocr_data["top"][i],
            ocr_data["width"][i],
            ocr_data["height"][i]]
        
        bboxes.append([(l/imgW)*1000, (t/imgH)*1000, ((l+w)/imgW)*1000, ((t+h)/imgH)*1000])
        ner_tags.append(ocr_data['label'][i].replace(" ",""))

        label = ocr_data['label'][i]

        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        


    labelJson["tokens"] = tokens
    labelJson["ner_tags"] = ner_tags
    labelJson["bboxes"] = bboxes

    with open(os.path.join(labelJsonFolder, f'{doc_id}.json'), 'w') as json_file:
        json.dump(labelJson, json_file, indent=4)


details_file = os.path.join(datasetFolder, "details.cfg")

with open(details_file, "w") as f:
    f.write("[General]\n")
    f.write(f"NumberOfSamples: {len(images_list)}\n\n")
    f.write("[LabelCounts]\n")
    for label in label_count:
        f.write(f"{label} = {label_count[label]}\n")