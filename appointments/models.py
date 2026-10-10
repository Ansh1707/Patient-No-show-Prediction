from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models
from prediction_service import FEATURES


def bounded_integer(low, high):
    return models.IntegerField(validators=[MinValueValidator(low), MaxValueValidator(high)])


class Appointment(models.Model):
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='appointments')
    Gender = models.CharField(max_length=1, choices=[('F', 'Female'), ('M', 'Male')])
    Age = bounded_integer(0, 115)
    Scholarship = bounded_integer(0, 1)
    Hipertension = bounded_integer(0, 1)
    Diabetes = bounded_integer(0, 1)
    Alcoholism = bounded_integer(0, 1)
    Handcap = bounded_integer(0, 4)
    SMS_received = bounded_integer(0, 1)
    wait_days = bounded_integer(-7, 365)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at', '-pk']
        constraints = [models.CheckConstraint(condition=models.Q(Gender__in=['F', 'M']), name='appointment_gender')]
        constraints += [models.CheckConstraint(condition=models.Q(**{f'{field}__gte': low, f'{field}__lte': high}),
                                               name=f'appointment_{field.lower()}_range')
                        for field, low, high in [('Age', 0, 115), ('Scholarship', 0, 1), ('Hipertension', 0, 1),
                                                ('Diabetes', 0, 1), ('Alcoholism', 0, 1), ('Handcap', 0, 4),
                                                ('SMS_received', 0, 1), ('wait_days', -7, 365)]]

    def features(self):
        return {name: getattr(self, name) for name in FEATURES}

    def __str__(self):
        return f'Appointment {self.pk} (user {self.owner_id})'


class PredictionHistory(models.Model):
    appointment = models.ForeignKey(Appointment, on_delete=models.CASCADE, related_name='predictions')
    requested_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True)
    input_snapshot = models.JSONField()
    probability_no_show = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    prediction = models.PositiveSmallIntegerField(choices=[(0, 'Show'), (1, 'No-show')])
    threshold = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    model_version = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at', '-pk']
        constraints = [models.CheckConstraint(condition=models.Q(probability_no_show__gte=0, probability_no_show__lte=1), name='prediction_probability_range'),
                       models.CheckConstraint(condition=models.Q(threshold__gte=0, threshold__lte=1), name='prediction_threshold_range'),
                       models.CheckConstraint(condition=models.Q(prediction__in=[0, 1]), name='prediction_binary')]

    def __str__(self):
        return f'Prediction {self.pk} for appointment {self.appointment_id}'
