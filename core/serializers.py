from rest_framework import serializers
from .models import DatabaseInstance, TableDescription, Query, Dashboard, PromptLog


class DatabaseInstanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatabaseInstance
        fields = '__all__'

class TableDescriptionSerializer(serializers.ModelSerializer):
    class Meta:
        model = TableDescription
        fields = '__all__'

class QuerySerializer(serializers.ModelSerializer):
    class Meta:
        model = Query
        fields = '__all__'

class DashboardSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dashboard
        fields = '__all__'

class PromptLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = PromptLog
        fields = '__all__'