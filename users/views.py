from rest_framework import viewsets, permissions
from .models import User, Organization
from .serializers import UserSerializer, OrganizationSerializer

class IsOrgAdminOrSuperUser(permissions.BasePermission):
    def has_permission(self, request, view):
        return request.user.is_superuser or (
            request.user.is_authenticated and request.user.is_organization_user and 
            request.user.organization and request.user in request.user.organization.organization_admins.all()
        )

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated, IsOrgAdminOrSuperUser]

    def get_queryset(self):
        user = self.request.user
        if user.is_superuser:
            return User.objects.all()
        return User.objects.filter(organization=user.organization)

class OrganizationViewSet(viewsets.ModelViewSet):
    queryset = Organization.objects.all()
    serializer_class = OrganizationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_superuser:
            return Organization.objects.all()
        return Organization.objects.filter(owner=user) | Organization.objects.filter(organization_admins=user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)