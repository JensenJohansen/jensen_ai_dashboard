# users/urls.py

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import UserViewSet, OrganizationViewSet

router = DefaultRouter()
router.register('users', UserViewSet, basename='users')
router.register('organizations', OrganizationViewSet, basename='organizations')

urlpatterns = [
    path('', include(router.urls)),
]
