# yourappname/views.py
from django.http import JsonResponse, HttpResponse
from django.core.files.storage import default_storage
from .models import SubTable, DataTable
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import SuspiciousOperation
from django.utils import timezone
from .utils import get_unique_id
import pytz
import base64
import magic
import json
import redis

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
channel = 'upload_doc'

def get_sub_table_data(request):
    ist = pytz.timezone('Asia/Kolkata')
    data = SubTable.objects.values('doc_id', 'status', 'inserted_time').order_by('-inserted_time')

    serialized_data = [
        {
            "doc_id": item['doc_id'],
            "status": item['status'],
            "inserted_time": timezone.localtime(item['inserted_time'], ist).strftime("%H:%M:%S %d-%m-%Y"),
        }
        for item in data
    ]
    return JsonResponse({"data": serialized_data})

def get_data_table_data(request, doc_id):
    try:
        # Retrieve data for the specified doc_id
        data_item = DataTable.objects.get(doc_id=doc_id)
        
        # Serialize the data
        serialized_data = {
            "doc_id": data_item.doc_id,
            "doc_json_ai": data_item.doc_json_ai,
            "doc_json_gt": data_item.doc_json_gt,
            "processed_time": timezone.localtime(data_item.processed_time, pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S %Z"),
            "human_verified_time": timezone.localtime(data_item.human_verified_time, pytz.timezone('UTC')).strftime("%Y-%m-%d %H:%M:%S %Z") if data_item.human_verified_time else None,
        }

        return JsonResponse(serialized_data)

    except DataTable.DoesNotExist:
        # Return a 404 response if the specified doc_id is not found
        return JsonResponse({"error": "Data not found for the given doc_id"}, status=404)

def get_document(request, doc_id: str):
    try:
        # Retrieve the SubTable record for the given doc_id
        sub_table_entry = SubTable.objects.get(doc_id=doc_id)

        # Decode the base64-encoded file content
        file_content = base64.b64decode(sub_table_entry.file)

        # Use magic library to determine file type
        mime = magic.Magic()
        file_type = mime.from_buffer(file_content)

        # Create a response with the file content and appropriate content type
        response = HttpResponse(file_content, content_type='application/octet-stream')

        # Adjust content type based on the detected file type
        if "pdf" in file_type.lower():
            response['Content-Type'] = 'application/pdf'
        elif "image" in file_type.lower():
            response['Content-Type'] = 'image'
        else:
            # You may want to handle other file types accordingly
            raise SuspiciousOperation('Unsupported file type')

        response['Content-Disposition'] = f'attachment; filename="{doc_id}"'

        return response

    except SubTable.DoesNotExist:
        return JsonResponse({'error': 'Document not found'}, status=404)


@csrf_exempt  # This decorator is used here for simplicity; consider using a more secure method for CSRF protection in production.
def upload_doc(request):
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')

        if uploaded_file:

            # Generate a unique ID for the document
            doc_id = get_unique_id()

            # Read the content of the uploaded file
            file_content = uploaded_file.read()

            # Encode the file content as base64
            file_base64 = base64.b64encode(file_content).decode('utf-8')

            # Store the file in the SubTable model
            SubTable.objects.create(
                doc_id=doc_id,
                file=file_base64,
                status='inqueue'
            )

            redis_client.publish(channel, doc_id)

            return JsonResponse({'message': 'File uploaded successfully'})
        else:
            return JsonResponse({'error': 'No file provided'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=405)

@csrf_exempt 
def save_data(request, doc_id):
    try:
        data = json.loads(request.body.decode('utf-8'))


        obj, created = DataTable.objects.update_or_create(
                            doc_id=doc_id,
                            defaults={
                                'doc_json_gt': data,
                                'human_verified_time': timezone.now(),
                            }
                        )

        SubTable.objects.filter(doc_id=doc_id).update(status='verified')

        return JsonResponse({'success': True})
    except json.JSONDecodeError as e:
        return JsonResponse({'error': 'Invalid JSON format'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
