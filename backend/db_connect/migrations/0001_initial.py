# Generated by Django 5.0.1 on 2024-01-21 19:17

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataTable',
            fields=[
                ('doc_id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('doc_json_ai', models.JSONField(null=True)),
                ('doc_json_gt', models.JSONField(null=True)),
                ('processed_time', models.DateTimeField(null=True)),
                ('human_verified_time', models.DateTimeField(null=True)),
            ],
            options={
                'db_table': 'data_table',
            },
        ),
        migrations.CreateModel(
            name='SubTable',
            fields=[
                ('doc_id', models.CharField(max_length=100, primary_key=True, serialize=False)),
                ('status', models.CharField(blank=True, max_length=100, null=True)),
                ('file', models.TextField(blank=True, null=True)),
                ('inserted_time', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'sub_table',
            },
        ),
    ]
