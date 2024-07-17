from django.urls import include, path
from .views import book

urlpatterns = [path("book/", book)]
