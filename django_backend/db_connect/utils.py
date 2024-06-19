from datetime import datetime
from .models import SubTable, Details

def get_unique_id():
    """ Create unique id for document using timestamp"""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")[:-3]


def save_file(file_path, uploaded_file):
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    