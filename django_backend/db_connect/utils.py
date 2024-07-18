from datetime import datetime
from .models import SubTable, Details

def get_unique_id():
    """ Create unique id for document using timestamp"""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")[:-3]


def save_file(destination_path, file_object):
    with open(destination_path, 'wb') as destination:
        for chunk in file_object.chunks():
            destination.write(chunk)


    