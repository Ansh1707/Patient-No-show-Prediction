from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter
from appointments.views import AppointmentViewSet, PredictionHistoryViewSet

router = DefaultRouter()
router.register('appointments', AppointmentViewSet, basename='appointment')
router.register('predictions', PredictionHistoryViewSet, basename='prediction')
urlpatterns = [path('admin/', admin.site.urls), path('api/', include(router.urls)),
               path('api-auth/', include('rest_framework.urls'))]
