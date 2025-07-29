from django.db.models.signals import pre_save, post_save
from django.dispatch import receiver
from common.middleware import get_current_user
from common.models import BaseModel, ActivityLog
import json

_previous_state = {}

def is_model_subclass(instance, cls):
    return isinstance(instance, cls) or (hasattr(instance, '__class__') and issubclass(instance.__class__, cls))

@receiver(pre_save)
def track_user_fields(sender, instance, **kwargs):
    from django.contrib.auth import get_user_model
    User = get_user_model()
    if is_model_subclass(instance, BaseModel):
        user = get_current_user()
        if user and isinstance(user, User):
            if not instance.pk and not instance.created_by:
                instance.created_by = user
            instance.updated_by = user

        # Capture previous state for diffs
        if instance.pk:
            try:
                original = sender.objects.get(pk=instance.pk)
                _previous_state[instance.pk] = {field.name: getattr(original, field.name) for field in instance._meta.fields if field.name != 'id'}
            except sender.DoesNotExist:
                pass

@receiver(post_save)
def log_model_save(sender, instance, created, **kwargs):
    if is_model_subclass(instance, BaseModel):
        action = 'create' if created else 'update'
        data_snapshot = {field.name: getattr(instance, field.name) for field in instance._meta.fields if field.name != 'id'}
        changes = {}
        if not created and instance.pk in _previous_state:
            before = _previous_state.pop(instance.pk)
            for k, v in data_snapshot.items():
                if before.get(k) != v:
                    changes[k] = {'from': before.get(k), 'to': v}

        ActivityLog.log_action(instance, action, extra_data={
            'changes': changes if changes else None
        })