# yourappname/urls.py
from django.urls import path
from .views import get_sub_table_data, get_data_table_data, get_document, upload_doc, save_data

urlpatterns = [
    path('sub_table/', get_sub_table_data, name='get_sub_table_data'),
    path('data_table/get_data/<str:doc_id>', get_data_table_data, name='get_data_table_data'),
    path('get-document/<str:doc_id>', get_document, name='get_document'),
    path('upload_doc/', upload_doc, name='upload_doc'),
    path('data_table/save_data/<str:doc_id>', save_data, name='save_data')
]