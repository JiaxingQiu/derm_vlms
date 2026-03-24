from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("annotations/", views.annotations_view, name="annotations"),
    path("thank_you/", views.thank_you_view, name="thank_you"),
    path("logout/", views.logout_view, name="logout"),
]