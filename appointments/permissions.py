from rest_framework.permissions import BasePermission


class OwnRecordOrStaff(BasePermission):
    def has_object_permission(self, request, view, obj):
        owner_id = obj.owner_id if hasattr(obj, 'owner_id') else obj.appointment.owner_id
        return request.user.is_staff or owner_id == request.user.pk
