import boto3
import os
import re
from tqdm import tqdm


# Set environment variable for AWS profile
os.environ['AWS_PROFILE'] = "Developers_custom1_tallydev-AI-381491826341"

# Create an STS client from the session
sts_client = boto3.client("sts")

# Assume the specified role
assumed_role_object = sts_client.assume_role(
    RoleArn="arn:aws:iam::891377125066:role/tally-ai-doc-ai-ml-engineer",
    RoleSessionName="AssumeRoleSession1",
)

# Extract temporary credentials
credentials = assumed_role_object["Credentials"]

# Use the assumed credentials to interact with S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=credentials["AccessKeyId"],
    aws_secret_access_key=credentials["SecretAccessKey"],
    aws_session_token=credentials["SessionToken"],
)


# S3 buckets
doc_bucket_name = "tally-imerit-data-safehouse"
json_bucket_name = "tally-imerit-jsons-safehouse"
individualFields_folder = ""
roi_folder = ""

current_dir = os.path.dirname(__file__)
dataset_folder = os.path.join(current_dir, "datasets", "imerit")

# Download files to a local folder
local_doc_folder = os.path.join(dataset_folder, "docs")
local_roi_json_folder = os.path.join(dataset_folder, "individualfield_jsons")

# Create directories if they don't exist
os.makedirs(local_doc_folder, exist_ok=True)
os.makedirs(local_roi_json_folder, exist_ok=True)


# Function to list all objects in an S3 bucket
def list_all_objects(s3_client, bucket_name):
    paginator = s3_client.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name)

    all_objects = []
    for response in response_iterator:
        if "Contents" in response:
            all_objects.extend(response["Contents"])
    return all_objects


# List all objects in the json bucket
json_objects = list_all_objects(s3_client, json_bucket_name)

n = 0

for i in tqdm(range(len(json_objects)), unit="sample"):
    # Extract object key
    obj = json_objects[i]
    json_obj_key = obj["Key"]
    # Check if the object is a file (not a "folder" or prefix)
    if not json_obj_key.endswith("/"):  # If the object key does not end with '/'
        # Define local file path

        json_filename = os.path.basename(json_obj_key)

        # Download the file
        if json_filename.find("IndividualField") != -1:
            local_json_path = os.path.join(local_roi_json_folder, json_filename)
            n += 1
            if os.path.exists(local_json_path):
                continue
        else:
            continue

        doc_filename_wo_ext = re.split("[._]", json_filename)[0]

        temp = json_obj_key.split("/")[:-1]
        temp.append(doc_filename_wo_ext)
        doc_obj_key_wo_ext = "/".join(temp)

        s3_client.download_file(json_bucket_name, json_obj_key, local_json_path)

        file_extentions = [".jpeg", ".png", ".jpg", ".pdf"]
        for ext in file_extentions:
            local_file_path = os.path.join(local_doc_folder, doc_filename_wo_ext + ext)
            try:
                s3_client.head_object(
                    Bucket=doc_bucket_name, Key=doc_obj_key_wo_ext + ext
                )
                s3_client.download_file(
                    doc_bucket_name, doc_obj_key_wo_ext + ext, local_file_path
                )
                break
            except:
                continue
if not json_objects:
    print("No objects found in the jsons bucket.")

print(f"Number of Samples: {n}")