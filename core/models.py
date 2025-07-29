from django.db import models
from common.models import BaseModel
from django.contrib.auth import get_user_model

User=get_user_model()

class DatabaseInstance(BaseModel):
    name = models.CharField(max_length=255)
    credentials = models.JSONField()
    description = models.TextField(blank=True, null=True)

class TableDescription(BaseModel):
    database = models.ForeignKey(DatabaseInstance, on_delete=models.CASCADE,related_name='tabledescription_set')
    table_name = models.CharField(max_length=255)
    column_descriptions = models.JSONField()
    description = models.TextField()

class Query(BaseModel):
    database = models.ForeignKey(DatabaseInstance, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    sql_output = models.TextField()
    failed = models.BooleanField(default=False)

class Dashboard(BaseModel):
    title = models.CharField(max_length=255)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    queries = models.ManyToManyField(Query)
    superset_id = models.IntegerField(null=True, blank=True)
    superset_url = models.URLField(null=True, blank=True)


class PromptLog(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    prompt = models.TextField()
    result = models.TextField(null=True, blank=True)
    failed = models.BooleanField(default=False)