from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_view, name="login"),
    path("annotations/", views.annotations_view, name="annotations"),
    path("logout/", views.logout_view, name="logout"),
]