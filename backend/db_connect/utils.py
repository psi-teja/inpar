from datetime import datetime

def get_unique_id():
    """ Create unique id for document using timestamp"""
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M%S%f")[:-3]