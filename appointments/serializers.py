from collections.abc import Mapping
from rest_framework import serializers
from .models import Appointment, PredictionHistory
from prediction_service import FEATURES


class AppointmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Appointment
        fields = ['id', 'owner', *FEATURES, 'created_at']
        read_only_fields = ['id', 'owner', 'created_at']

    def to_internal_value(self, data):
        if not isinstance(data, Mapping):
            return super().to_internal_value(data)
        unexpected = set(data) - set(FEATURES)
        if unexpected:
            raise serializers.ValidationError({key: 'This field cannot be supplied.' for key in unexpected})
        # Require actual JSON integers; reject booleans and fractional/string coercion.
        errors = {key: 'An integer is required.' for key in FEATURES[1:]
                  if key in data and type(data[key]) is not int}
        if errors:
            raise serializers.ValidationError(errors)
        return super().to_internal_value(data)


class PredictionHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionHistory
        fields = ['id', 'appointment', 'requested_by', 'input_snapshot', 'probability_no_show',
                  'prediction', 'threshold', 'model_version', 'created_at']
        read_only_fields = fields
