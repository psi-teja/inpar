from django.urls import path
from .views import get_sub_table_data, get_data_table_data, get_document, upload_doc, save_data

urlpatterns = [
    path('get/SubTable/', get_sub_table_data, name='get_sub_table_data'),
    path('get/json/<str:json_type>/<str:doc_id>/', get_data_table_data, name='get_data_table_data'),
    path('get/document/<str:doc_id>/', get_document, name='get_document'),
    path('upload/doc/', upload_doc, name='upload_doc'),
    path('upload/json/<str:json_type>/<str:doc_id>/', save_data, name='save_data')  # Ensure the trailing slash is present
]