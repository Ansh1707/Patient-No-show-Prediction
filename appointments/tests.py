import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from django.contrib.auth import get_user_model
from django.db import IntegrityError, transaction
from django.test import SimpleTestCase
from rest_framework.authtoken.models import Token
from rest_framework.test import APITestCase, APIClient
from fastapi.testclient import TestClient

from prediction_service import FEATURES, PredictionService, PredictionUnavailable
from .models import Appointment, PredictionHistory

INPUTS = dict(Gender='F', Age=30, Scholarship=0, Hipertension=0, Diabetes=0,
              Alcoholism=0, Handcap=0, SMS_received=1, wait_days=2)


class DeterministicModel:
    """Serialized test double, never a replacement for the production model."""
    def predict_proba(self, data):
        assert list(data.columns) == list(FEATURES)
        probability = 0.3036 if data.iloc[0]['Age'] == 30 else 0.8
        return np.array([[1 - probability, probability]])


class ServiceTests(SimpleTestCase):
    def test_saved_artifacts_fallback_threshold_and_exact_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model = path / 'model.pkl'
            joblib.dump(DeterministicModel(), model)
            service = PredictionService(model, path / 'missing.json')
            result = service.predict(INPUTS)
            self.assertEqual(result.prediction, 0)  # 0.304 rounded, but raw score is below threshold.
            self.assertEqual(result.response(), {'probability_no_show': 0.304, 'prediction': 0})
            self.assertEqual(len(result.model_version), 64)
            (path / 'threshold.json').write_text(json.dumps({'threshold': 0.3036}))
            tuned = PredictionService(model, path / 'threshold.json')
            self.assertEqual(tuned.predict(INPUTS).prediction, 1)
            self.assertEqual(tuned.model_version, service.model_version)

    def test_missing_and_corrupt_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(PredictionUnavailable):
                PredictionService(path / 'missing.pkl', path / 'threshold.json')
            joblib.dump(DeterministicModel(), path / 'model.pkl')
            for value in [-1, 2, 'bad', float('nan')]:
                (path / 'threshold.json').write_text(json.dumps({'threshold': value}))
                with self.subTest(value=value), self.assertRaises(PredictionUnavailable):
                    PredictionService(path / 'model.pkl', path / 'threshold.json')

    def test_invalid_model_output_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            joblib.dump(DeterministicModel(), path / 'model.pkl')
            service = PredictionService(path / 'model.pkl', path / 'none.json')
            for value in [-0.1, 1.1, float('nan'), float('inf')]:
                with patch.object(service.model, 'predict_proba', return_value=[[0, value]]):
                    with self.subTest(value=value), self.assertRaises(PredictionUnavailable):
                        service.predict(INPUTS)


class AppointmentAPITests(APITestCase):
    def setUp(self):
        users = get_user_model()
        self.owner = users.objects.create_user('owner', password='password-for-tests')
        self.other = users.objects.create_user('other', password='password-for-tests')
        self.staff = users.objects.create_user('staff', password='password-for-tests', is_staff=True)
        self.appointment = Appointment.objects.create(owner=self.owner, **INPUTS)
        self.other_appointment = Appointment.objects.create(owner=self.other, **INPUTS)
        self.url = f'/api/appointments/{self.appointment.pk}/'
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name)
        joblib.dump(DeterministicModel(), path / 'model.pkl')
        self.service = PredictionService(path / 'model.pkl', path / 'missing.json')
        patcher = patch('appointments.views.get_prediction_service', return_value=self.service)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client.force_authenticate(self.owner)

    def test_create_and_retrieve_appointment(self):
        response = self.client.post('/api/appointments/', INPUTS, format='json')
        self.assertEqual(response.status_code, 201)
        saved = Appointment.objects.get(pk=response.data['id'])
        self.assertEqual(saved.owner, self.owner)
        self.assertEqual(saved.features(), INPUTS)
        self.assertEqual(self.client.get(f'/api/appointments/{saved.pk}/').data['Age'], 30)

    def test_invalid_inputs_do_not_create_records(self):
        invalid = [('Age', -1), ('Age', 116), ('Gender', 'X'), ('wait_days', -8),
                   ('wait_days', 366), ('Handcap', -1), ('Handcap', 5), ('Age', 1.5),
                   ('Age', True), ('Age', '30'), ('Age', None)]
        invalid += [(field, value) for field in ('Scholarship', 'Hipertension', 'Diabetes',
                                                'Alcoholism', 'SMS_received') for value in (-1, 2)]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                response = self.client.post('/api/appointments/', {**INPUTS, field: value}, format='json')
                self.assertEqual(response.status_code, 400)
        for field in INPUTS:
            data = INPUTS.copy()
            del data[field]
            with self.subTest(missing=field):
                self.assertEqual(self.client.post('/api/appointments/', data, format='json').status_code, 400)
        self.assertEqual(Appointment.objects.count(), 2)

    def test_ownership_and_extra_fields_cannot_be_supplied(self):
        for field, value in [('owner', self.other.pk), ('id', 100), ('unknown', 1)]:
            with self.subTest(field=field):
                self.assertEqual(self.client.post('/api/appointments/', {**INPUTS, field: value}, format='json').status_code, 400)

    def test_valid_boundaries(self):
        for age, wait, handicap in [(0, -7, 0), (115, 365, 4)]:
            self.assertEqual(self.client.post('/api/appointments/',
                {**INPUTS, 'Age': age, 'wait_days': wait, 'Handcap': handicap}, format='json').status_code, 201)

    def test_anonymous_requests_cannot_read_create_or_predict(self):
        self.client.force_authenticate(None)
        requests = [('get', '/api/appointments/', None), ('get', self.url, None),
                    ('post', '/api/appointments/', INPUTS), ('post', self.url + 'predict/', {}),
                    ('get', self.url + 'history/', None), ('get', '/api/predictions/', None)]
        for method, url, data in requests:
            with self.subTest(method=method, url=url):
                self.assertEqual(getattr(self.client, method)(url, data, format='json').status_code, 401)
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_other_users_cannot_retrieve_predict_or_list_records(self):
        self.client.post(self.url + 'predict/', {}, format='json')
        prediction = PredictionHistory.objects.get()
        self.client.force_authenticate(self.other)
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.client.post(self.url + 'predict/', {}, format='json').status_code, 404)
        self.assertEqual(self.client.get(self.url + 'history/').status_code, 404)
        self.assertEqual(self.client.get(f'/api/predictions/{prediction.pk}/').status_code, 404)
        self.assertEqual(self.client.get('/api/predictions/').data['count'], 0)
        listed = self.client.get('/api/appointments/').data['results']
        self.assertEqual([row['id'] for row in listed], [self.other_appointment.pk])
        self.assertEqual(PredictionHistory.objects.count(), 1)

    def test_staff_can_review_and_predict_all_records(self):
        self.client.force_authenticate(self.staff)
        self.assertEqual(self.client.get('/api/appointments/').data['count'], 2)
        self.assertEqual(self.client.get(self.url).status_code, 200)
        response = self.client.post(self.url + 'predict/', {}, format='json')
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data['requested_by'], self.staff.pk)
        self.assertEqual(self.client.get('/api/predictions/').data['count'], 1)
        self.assertEqual(self.client.get(self.url + 'history/').data['count'], 1)
        self.assertEqual(Appointment.objects.get(pk=self.appointment.pk).owner, self.owner)

    def test_prediction_is_saved_and_retrieved_with_exact_inputs(self):
        expected = self.service.predict(INPUTS)
        for _ in range(2):
            response = self.client.post(self.url + 'predict/', {}, format='json')
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.data['probability_no_show'], expected.probability_no_show)
            self.assertEqual(response.data['prediction'], expected.prediction)
        saved = PredictionHistory.objects.first()
        self.assertEqual(saved.input_snapshot, INPUTS)
        self.assertEqual(saved.threshold, expected.threshold)
        self.assertEqual(saved.model_version, expected.model_version)
        self.assertEqual(saved.requested_by, self.owner)
        self.assertEqual(self.client.get(self.url + 'history/').data['count'], 2)
        detail = self.client.get(f'/api/predictions/{saved.pk}/')
        self.assertEqual(detail.data['input_snapshot'], INPUTS)
        # History preserves the prediction inputs even if staff edit an appointment later.
        self.appointment.Age = 60
        self.appointment.save()
        saved.refresh_from_db()
        self.assertEqual(saved.input_snapshot['Age'], 30)

    def test_prediction_consistency_between_fastapi_and_django(self):
        from app import app
        with patch('app.get_prediction_service', return_value=self.service):
            legacy = TestClient(app).post('/predict', json=INPUTS)
        django = self.client.post(self.url + 'predict/', {}, format='json')
        self.assertEqual(legacy.status_code, 200)
        self.assertEqual(legacy.json()['probability_no_show'], round(django.data['probability_no_show'], 3))
        self.assertEqual(legacy.json()['prediction'], django.data['prediction'])

    def test_unavailable_service_saves_nothing(self):
        with self.assertLogs('appointments.views', level='WARNING'), patch(
                'appointments.views.get_prediction_service', side_effect=PredictionUnavailable('private path')):
            response = self.client.post(self.url + 'predict/', {}, format='json')
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private path', str(response.data))
        self.assertEqual(PredictionHistory.objects.count(), 0)

    def test_cannot_override_prediction_inputs_or_modify_history(self):
        self.assertEqual(self.client.post(self.url + 'predict/', {'Age': 80}, format='json').status_code, 400)
        self.client.post(self.url + 'predict/', {}, format='json')
        saved = PredictionHistory.objects.get()
        self.assertEqual(self.client.post('/api/predictions/', {}, format='json').status_code, 405)
        for method in ('patch', 'delete'):
            self.assertEqual(getattr(self.client, method)(f'/api/predictions/{saved.pk}/', {}, format='json').status_code, 405)

    def test_token_authentication_and_bad_token(self):
        self.client.force_authenticate(None)
        token = Token.objects.create(user=self.owner)
        self.client.credentials(HTTP_AUTHORIZATION=f'Token {token.key}')
        self.assertEqual(self.client.get(self.url).status_code, 200)
        self.client.credentials(HTTP_AUTHORIZATION='Token invalid')
        self.assertEqual(self.client.get(self.url).status_code, 401)

    def test_database_enforces_ranges(self):
        with self.assertRaises(IntegrityError), transaction.atomic():
            Appointment.objects.create(owner=self.owner, **{**INPUTS, 'Age': -1})

    def test_admin_lists_records_and_history_is_read_only(self):
        users = get_user_model()
        admin = users.objects.create_superuser('admin', password='password-for-tests')
        self.client.post(self.url + 'predict/', {}, format='json')
        self.client.force_login(admin)
        self.assertEqual(self.client.get('/admin/appointments/appointment/').status_code, 200)
        self.assertEqual(self.client.get('/admin/appointments/predictionhistory/').status_code, 200)
        self.assertEqual(self.client.get('/admin/appointments/predictionhistory/add/').status_code, 403)

    def test_non_object_request_bodies_are_validation_errors(self):
        for body in ([INPUTS], ['Age'], 'invalid', 42):
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/api/appointments/', body, format='json').status_code, 400)

    def test_session_authentication_requires_csrf_for_writes(self):
        client = APIClient(enforce_csrf_checks=True)
        client.force_login(self.owner)
        self.assertEqual(client.get(self.url).status_code, 200)
        self.assertEqual(client.post('/api/appointments/', INPUTS, format='json').status_code, 403)
        # Visiting the browsable API login form sets the CSRF cookie.
        client.get('/api-auth/login/')
        csrf = client.cookies['csrftoken'].value
        self.assertEqual(client.post('/api/appointments/', INPUTS, format='json',
                                     HTTP_X_CSRFTOKEN=csrf).status_code, 201)

    def test_real_sklearn_pipeline_matches_direct_inference_and_saved_history(self):
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        rows = [{**INPUTS, 'Gender': 'F' if n % 2 else 'M', 'Age': n + 10} for n in range(40)]
        frame = pd.DataFrame(rows, columns=FEATURES)
        pipeline = Pipeline([
            ('prep', ColumnTransformer([('gender', OneHotEncoder(handle_unknown='ignore'), ['Gender'])],
                                       remainder='passthrough')),
            ('clf', LogisticRegression(random_state=42))])
        pipeline.fit(frame, [int(n >= 20) for n in range(40)])
        path = Path(self.directory.name)
        joblib.dump(pipeline, path / 'real.pkl')
        (path / 'threshold.json').write_text(json.dumps({'threshold': 0.42}))
        service = PredictionService(path / 'real.pkl', path / 'threshold.json')
        with patch('appointments.views.get_prediction_service', return_value=service):
            response = self.client.post(self.url + 'predict/', {}, format='json')
        probability = float(pipeline.predict_proba(pd.DataFrame([INPUTS], columns=FEATURES))[0][1])
        self.assertEqual(response.status_code, 201)
        self.assertAlmostEqual(response.data['probability_no_show'], probability)
        self.assertEqual(response.data['prediction'], int(probability >= 0.42))
        saved = PredictionHistory.objects.get()
        self.assertAlmostEqual(saved.probability_no_show, probability)
        self.assertEqual(saved.input_snapshot, INPUTS)

    def test_form_encoded_inputs_are_rejected_use_json_instead(self):
        response = self.client.post('/api/appointments/', INPUTS, format='multipart')
        self.assertEqual(response.status_code, 415)
        self.assertEqual(Appointment.objects.count(), 2)
