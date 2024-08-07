# yourappname/views.py
from django.http import JsonResponse, HttpResponse, Http404
from django.core.files.storage import default_storage
from .models import SubTable, Details
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
from .utils import get_unique_id, save_file
import pytz
from django.conf import settings
from django.db import transaction
from PIL import Image
import json
import redis
import os
import io
from datetime import datetime
import logging, shutil
import pytesseract

# Initialize Redis client
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = 'upload_doc'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@csrf_exempt 
def get_sub_table_data(request):
    ist = pytz.timezone('Asia/Kolkata')
    data = SubTable.objects.values("doc_id", "status", "s3_file", "local_file", 'inserted_time').order_by('-inserted_time')

    serialized_data = [
        {
            "doc_id": item['doc_id'],
            "status": item['status'],
            "s3_file": item['s3_file'],
            "local_file": item['local_file'],
            "inserted_time": timezone.localtime(item['inserted_time'], ist).strftime("%H:%M:%S %d-%m-%Y"),
        }
        for item in data
    ]
    return JsonResponse(serialized_data, safe=False)

def get_data_table_data(request, json_type: str, doc_id: str):
    json_dir = os.path.join(settings.BASE_DIR, 'output_jsons')
    json_path = os.path.join(json_dir, json_type, f"{doc_id}.json")

    if not os.path.exists(json_path):
        raise Http404("JSON file not found")

    with open(json_path, "r") as f:
        output_json = json.load(f)

    return JsonResponse(output_json, safe=False)

def get_document(request, doc_id: str):
    try:
        item = SubTable.objects.values("s3_file").get(doc_id=doc_id)
        upload_dir = os.path.join(settings.BASE_DIR, 'input_docs')
        s3_filename = os.path.join(upload_dir, item['s3_file'])

        if not os.path.exists(s3_filename):
            raise Http404("Document file not found")

        with open(s3_filename, 'rb') as file:
            file_content = file.read()

        if not s3_filename.endswith('.pdf'):
            img = Image.open(io.BytesIO(file_content))
            pdf_bytes = io.BytesIO()
            img.save(pdf_bytes, format='PDF')
            pdf_bytes.seek(0)
            file_content = pdf_bytes.read()

        # Set headers for PDF content
        response = HttpResponse(file_content, content_type='application/pdf')
        response['Content-Disposition'] = f'attachment; filename="{doc_id}.pdf"'
        return response

    except SubTable.DoesNotExist:
        logger.error(f"Document with doc_id {doc_id} does not exist.")
        raise Http404("Document not found")
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise Http404(str(e))

@csrf_exempt 
def upload_doc(request):
    if request.method == 'POST':
        if 'document' not in request.FILES:
            return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)

        uploaded_file = request.FILES['document']

        # Generate a unique ID for the document
        doc_id = get_unique_id()

        # Define the directory where you want to save the file
        upload_dir = os.path.join(settings.BASE_DIR, 'input_docs')

        # Create the directory if it doesn't exist
        os.makedirs(upload_dir, exist_ok=True)

        filename = f'{doc_id}.{uploaded_file.name.split(".")[-1]}'
        file_path = os.path.join(upload_dir, filename)

        # Save the file asynchronously
        save_file(file_path, uploaded_file)

        SubTable.objects.create(
            doc_id=doc_id,
            s3_file=filename,
            local_file=uploaded_file.name,
            status="inqueue"
        )
        # Create the request dictionary
        request_dict = {"doc_id": doc_id, "file_path": file_path}

        # Convert the dictionary to a JSON string
        request_json = json.dumps(request_dict)

        redis_client.publish(channel, request_json)

        return JsonResponse({'status': 'success', 'file_path': file_path})
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)

@csrf_exempt 
def save_data(request, json_type: str, doc_id: str):
    try:
        data = json.loads(request.body.decode('utf-8'))

        # Construct file path securely
        file_path = os.path.join(settings.BASE_DIR, 'output_jsons', json_type, f'{doc_id}.json')

        if json_type == "ai_json":
            updated_status = "processed"
        elif json_type == "gt_json":
            updated_status = "verified"

        if not data:
            updated_status = "failed"

        # Create directories if they do not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save data to JSON file
        with open(file_path, 'w') as f:
            json.dump(data, f)

        # Update SubTable status within a transaction
        with transaction.atomic():
            SubTable.objects.filter(doc_id=doc_id).update(status=updated_status)

        return JsonResponse({'success': True})
    except json.JSONDecodeError:
        logger.error("Invalid JSON format received.")
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        return JsonResponse({'error': str(e)}, status=500)



@csrf_exempt
def upload_pre_labels(request):

    purpose = "addLedger"

    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            images_folder = data.get('images_folder')
            jsons_folder = data.get('jsons_folder')

            if not images_folder or not jsons_folder:
                return JsonResponse({'status': 'error', 'message': 'Folders paths not provided'}, status=400)

            # Ensure the folders exist
            if not os.path.exists(images_folder) or not os.path.exists(jsons_folder):
                return JsonResponse({'status': 'error', 'message': 'Folders do not exist'}, status=400)

            # Process each image in the images folder
            for image_filename in os.listdir(images_folder):
                if image_filename.endswith(('png', 'jpg', 'jpeg')):
                    image_path = os.path.join(images_folder, image_filename)
                    doc_id = os.path.splitext(image_filename)[0]

                    # Corresponding JSON file path
                    json_filename = doc_id + '.json'
                    json_path = os.path.join(jsons_folder, json_filename)

                    if not os.path.exists(json_path):
                        logger.warning(f"JSON file for {image_filename} not found, skipping.")
                        continue

                    # Save the image file
                    upload_dir = os.path.join(settings.BASE_DIR, 'input_docs')
                    os.makedirs(upload_dir, exist_ok=True)
                    s3_filename = f'{doc_id}.{image_filename.split(".")[-1]}'
                    file_path = os.path.join(upload_dir, s3_filename)

                    shutil.copy(image_path, file_path)

                    # Save data to SubTable
                    SubTable.objects.create(
                        doc_id=doc_id,
                        s3_file=s3_filename,
                        local_file=image_filename,
                        status=purpose
                    )

                    output_json_dir = os.path.join(settings.BASE_DIR, 'output_jsons', purpose)
                    os.makedirs(output_json_dir, exist_ok=True)
                    output_json_path = os.path.join(output_json_dir, f'{doc_id}.json')

                    shutil.copy(json_path, output_json_path)

                    # Update status to processed
                    # SubTable.objects.filter(doc_id=doc_id).update(status="processed")

            return JsonResponse({'status': 'success', 'message': 'Pre-labeled data uploaded successfully'})
        except json.JSONDecodeError:
            logger.error("Invalid JSON format received.")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            logger.error(f"Error uploading pre-labeled data: {e}")
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'}, status=400)


@csrf_exempt
def get_ocr_text(request):
    if request.method == 'POST' and request.FILES['file']:
        uploaded_file = request.FILES['file']

        # Use Pillow to open the uploaded image file
        image = Image.open(uploaded_file)

        # Apply OCR using pytesseract
        text = pytesseract.image_to_string(image)

        # Return the extracted text as JSON response
        return JsonResponse({'text': text})

    return JsonResponse({'error': 'POST method and file data required'})