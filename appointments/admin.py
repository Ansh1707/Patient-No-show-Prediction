from django.contrib import admin
from .models import Appointment, PredictionHistory


@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ['id', 'owner', 'Gender', 'Age', 'wait_days', 'created_at']
    list_filter = ['Gender', 'SMS_received', 'created_at']
    search_fields = ['owner__username']
    list_select_related = ['owner']


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'appointment', 'requested_by', 'prediction', 'probability_no_show', 'created_at']
    list_filter = ['prediction', 'created_at']
    search_fields = ['appointment__owner__username', 'model_version']
    list_select_related = ['appointment', 'requested_by']
    readonly_fields = [field.name for field in PredictionHistory._meta.fields]

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
