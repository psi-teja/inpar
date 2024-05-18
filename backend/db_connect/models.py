from django.db import models

class SubTable(models.Model):
    doc_id = models.CharField(max_length=100, primary_key=True)
    status = models.CharField(max_length=100, blank=True, null=True)
    file = models.TextField(blank=True, null=True)
    inserted_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'sub_table'

class DataTable(models.Model):
    doc_id = models.CharField(max_length=100, primary_key=True)
    doc_json_ai = models.JSONField(null=True)
    doc_json_gt = models.JSONField(null=True)
    processed_time = models.DateTimeField(null=True)
    human_verified_time = models.DateTimeField(null=True)

    class Meta:
        db_table = 'data_table'
