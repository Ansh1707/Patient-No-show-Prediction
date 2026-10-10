import logging
from django.db import transaction
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from prediction_service import get_prediction_service, PredictionUnavailable
from .models import Appointment, PredictionHistory
from .permissions import OwnRecordOrStaff
from .serializers import AppointmentSerializer, PredictionHistorySerializer

logger = logging.getLogger(__name__)


class AppointmentViewSet(mixins.CreateModelMixin, mixins.ListModelMixin,
                         mixins.RetrieveModelMixin, viewsets.GenericViewSet):
    serializer_class = AppointmentSerializer
    permission_classes = [IsAuthenticated, OwnRecordOrStaff]

    def get_queryset(self):
        queryset = Appointment.objects.select_related('owner')
        return queryset if self.request.user.is_staff else queryset.filter(owner=self.request.user)

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

    @action(detail=True, methods=['post'])
    def predict(self, request, pk=None):
        appointment = self.get_object()
        if request.data:
            return Response({'detail': 'Prediction uses saved appointment inputs; send an empty body.'}, status=400)
        inputs = appointment.features()
        try:
            result = get_prediction_service().predict(inputs)
        except PredictionUnavailable:
            logger.warning('Prediction service unavailable', exc_info=True)
            return Response({'detail': 'Prediction service is unavailable. Please try again later.'}, status=503)
        with transaction.atomic():
            saved = PredictionHistory.objects.create(
                appointment=appointment, requested_by=request.user, input_snapshot=inputs,
                probability_no_show=result.probability_no_show, prediction=result.prediction,
                threshold=result.threshold, model_version=result.model_version)
        return Response(PredictionHistorySerializer(saved).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=['get'])
    def history(self, request, pk=None):
        queryset = self.get_object().predictions.all()
        page = self.paginate_queryset(queryset)
        return self.get_paginated_response(PredictionHistorySerializer(page, many=True).data)


class PredictionHistoryViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = PredictionHistorySerializer
    permission_classes = [IsAuthenticated, OwnRecordOrStaff]

    def get_queryset(self):
        queryset = PredictionHistory.objects.select_related('appointment', 'requested_by')
        return queryset if self.request.user.is_staff else queryset.filter(appointment__owner=self.request.user)
