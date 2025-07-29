import threading

_user = threading.local()

def get_current_user():
    return getattr(_user, 'value', None)

def set_current_user(user):
    _user.value = user

class CurrentUserMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, 'user', None)
        set_current_user(user if getattr(user, 'is_authenticated', False) else None)
        response = self.get_response(request)
        return response
