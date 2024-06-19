# %%
import boto3
import json
import os

# %%
textract = boto3.client(
    "textract",
    region_name="ap-south-1",
    aws_access_key_id="",
    aws_secret_access_key="",
)


# %%
def image_texts_bboxes(image_path):
    with open(image_path, "rb") as document:
        imageBytes = document.read()
    response = textract.detect_document_text(Document={"Bytes": imageBytes})
    texts = []
    bboxes = []
    for item in response["Blocks"]:
        if item["BlockType"] == "LINE":
            texts.append(item["Text"])
            bboxes.append(
                [
                    item["Geometry"]["BoundingBox"]["Left"],
                    item["Geometry"]["BoundingBox"]["Top"],
                    item["Geometry"]["BoundingBox"]["Width"],
                    item["Geometry"]["BoundingBox"]["Height"],
                ]
            )
    texts_bboxs = {"texts": texts, "bboxes_ltwh": bboxes}

    with open(
        os.path.join(
            "extracted_texts_bboxs", image_path.split("\\")[-1].split(".")[0] + ".json"
        ),
        "w",
    ) as fp:
        json.dump(texts_bboxs, fp)


# %%
list_dir = os.listdir("data")
for dir in list_dir:
    image_path = os.path.join("data", dir)
    image_texts_bboxes(image_path)

# %%
