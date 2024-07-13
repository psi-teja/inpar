import random
import boto3
import os
import zipfile
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Set environment variable for AWS profile
os.environ["AWS_PROFILE"] = "psi.teja_aws_profile"
# Define S3 bucket and model name
model_artifacts_bucket = "model-artifacts"
model_name = "yolov5-table"

def get_random_color():
    """Generate a random color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def download_model_from_s3(job_dir):
    job_id = os.path.basename(job_dir)
    os.makedirs(job_dir, exist_ok=True)
    s3_client = boto3.client('s3')
    zip_filename = f"{job_id}.zip"
    zip_file_path = job_dir+".zip"
    s3_key = f"{model_name}/{zip_filename}"
    try:
        s3_client.download_file(model_artifacts_bucket, s3_key, zip_file_path)
        print(f"Downloaded {zip_file_path} from S3 bucket {model_artifacts_bucket}/{s3_key}")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(job_dir)
        print(f"Extracted {zip_file_path} to {job_dir}")
        
        # Remove the zip file after extraction
        os.remove(zip_file_path)
        print(f"Deleted zip file {zip_file_path} after extraction")
        
        return True

    except FileNotFoundError:
        print(f"The file {zip_file_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

    return False

def zip_directory(folder_path, zip_file_path):
    """Zip the contents of an entire directory."""
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    print(f"Directory {folder_path} zipped into {zip_file_path}")

def upload_file_to_s3(folder_path, model_name):
    """Upload a file to an S3 bucket."""
    s3_client = boto3.client('s3')

    zip_file_path = folder_path+".zip"

    zip_filename = os.path.basename(zip_file_path)

    zip_directory(folder_path, zip_file_path)

    s3_key = f"{model_name}/{zip_filename}"

    try:
        s3_client.upload_file(zip_file_path, model_artifacts_bucket, s3_key)
        print(f"File {zip_file_path} uploaded to {model_artifacts_bucket}/{s3_key}")

        os.remove(zip_file_path)
        print(f"File {zip_file_path} deleted after upload")

    except FileNotFoundError:
        print(f"The file {zip_file_path} was not found")
    except NoCredentialsError:
        print("Credentials not available")
    except PartialCredentialsError:
        print("Incomplete credentials provided")
    except Exception as e:
        print(f"An error occurred: {str(e)}")