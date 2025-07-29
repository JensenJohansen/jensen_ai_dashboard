from django.db import models
from django.utils import timezone
from django.conf import settings
import uuid

class UUIDModel(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    class Meta:
        abstract = True

class TimeStampedModel(models.Model):
    created_at = models.DateTimeField(default=timezone.now, editable=False)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class SoftDeleteModel(models.Model):
    is_deleted = models.BooleanField(default=False)

    class Meta:
        abstract = True

    def delete(self, using=None, keep_parents=False):
        self.is_deleted = True
        self.save(update_fields=['is_deleted'])
        ActivityLog.log_action(self, 'soft_delete')

    def restore(self):
        self.is_deleted = False
        self.save(update_fields=['is_deleted'])
        ActivityLog.log_action(self, 'restore')

class TrailModel(models.Model):
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="created_%(class)s_set",
        on_delete=models.SET_NULL, null=True, blank=True
    )
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, related_name="updated_%(class)s_set",
        on_delete=models.SET_NULL, null=True, blank=True
    )

    class Meta:
        abstract = True

class BaseModel(UUIDModel, TimeStampedModel, SoftDeleteModel, TrailModel):
    class Meta:
        abstract = True

class ActivityLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=100)
    model_name = models.CharField(max_length=100)
    object_id = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    data_snapshot = models.JSONField(null=True, blank=True)
    extra_data = models.JSONField(null=True, blank=True)

    @classmethod
    def log_action(cls, instance, action, user=None, extra_data=None):
        from common.middleware import get_current_user
        if not user:
            user = get_current_user()
        cls.objects.create(
            user=user,
            action=action,
            model_name=instance.__class__.__name__,
            object_id=str(instance.pk),
            data_snapshot={field.name: getattr(instance, field.name) for field in instance._meta.fields if field.name != 'id'},
            extra_data=extra_data
        )