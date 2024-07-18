import random
import os
import zipfile

# Set environment variable for AWS profile
os.environ["AWS_PROFILE"] = "psi.teja_aws_profile"
# Define S3 bucket and model name
model_artifacts_bucket = "model-artifacts"
model_name = "yolov5-table"

def get_random_color():
    """Generate a random color."""
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


