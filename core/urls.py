from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    DatabaseInstanceViewSet,
    QueryViewSet,
    DashboardViewSet,
    PromptLogViewSet,
)

router = DefaultRouter()
router.register('databases', DatabaseInstanceViewSet, basename='databases')
router.register('queries', QueryViewSet, basename='queries')
router.register('dashboards', DashboardViewSet, basename='dashboards')
router.register('prompts', PromptLogViewSet, basename='prompts')

urlpatterns = [
    path('', include(router.urls)),
]
