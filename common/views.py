from django.shortcuts import render
from rest_framework import viewsets, permissions
from .serializers import ActivityLogSerializer
from .models import ActivityLog

class ActivityLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = ActivityLog.objects.all().order_by('-timestamp')
    serializer_class = ActivityLogSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        queryset = super().get_queryset()
        show_deleted = self.request.query_params.get('include_deleted') == 'true'
        if not show_deleted:
            queryset = queryset.exclude(data_snapshot__is_deleted=True)
        return queryset