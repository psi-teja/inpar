from django.db import models

class SubTable(models.Model):
    doc_id = models.CharField(max_length=100, primary_key=True)
    status = models.CharField(max_length=100, null=True, blank=True)
    s3_file = models.TextField(null=True, blank=True)
    local_file = models.TextField(null=True, blank=True)
    inserted_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'SubTable'
        ordering = ['-inserted_time']

    def __str__(self):
        return self.doc_id

class Details(models.Model):
    filename = models.CharField(max_length=255)
    uuid = models.CharField(max_length=100, primary_key=True)
    database_json = models.CharField(max_length=255, null=True, blank=True)
    status = models.CharField(max_length=50, null=True, blank=True)

    class Meta:
        db_table = 'details'

    def __str__(self):
        return self.uuid
