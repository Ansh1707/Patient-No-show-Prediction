# Patient No-Show Prediction: Django appointment management

Django now provides authenticated appointment management and persistent prediction history
around the existing trained pipeline. The original FastAPI demo remains available.
See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the completed step-by-step implementation.

## Run Django locally

Python 3.11 is used by CI. From the repository directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DJANGO_DEBUG=true
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Open `http://127.0.0.1:8000/admin/` to inspect appointments and saved predictions.
Use `http://127.0.0.1:8000/api-auth/login/` to log in, then open
`http://127.0.0.1:8000/api/appointments/` for the browsable API. Submit JSON using its Raw data tab. Session POSTs require CSRF.
Create ordinary users in admin. Staff can review all records through the API; give staff the
appropriate Django model permissions (or use a superuser) to access records in admin.

For a script or API client, issue a token for an existing user:

```sh
python manage.py drf_create_token username
```

Send it as `Authorization: Token <token>`. Use HTTPS outside local development. Tokens are
issued by the operator, so this version does not expose registration or an unauthenticated
token issuance endpoint. Revoke tokens through the token admin.

## Django endpoints

All routes require authentication. Regular users see only their own records; staff see all.

| Method | Route | Behavior |
| --- | --- | --- |
| POST | `/api/appointments/` | Create an appointment owned by the signed-in user |
| GET | `/api/appointments/` | Paginated appointment list |
| GET | `/api/appointments/{id}/` | Appointment detail |
| POST | `/api/appointments/{id}/predict/` | Predict saved inputs and save a history row; body `{}` |
| GET | `/api/appointments/{id}/history/` | Paginated history for one appointment |
| GET | `/api/predictions/` | Paginated accessible prediction history |
| GET | `/api/predictions/{id}/` | Saved prediction detail |

Create an appointment with these exact model feature names:

```json
{
  "Gender": "F", "Age": 30, "Scholarship": 0, "Hipertension": 0,
  "Diabetes": 0, "Alcoholism": 0, "Handcap": 0, "SMS_received": 1,
  "wait_days": 2
}
```

Use `Gender` F/M, age 0..115, binary flags 0/1, handicap 0..4 and waiting days -7..365.
Numeric inputs must be JSON integers. Missing, extra and invalid fields return 400. Ownership
is assigned by the server. Unauthenticated requests return 401; inaccessible IDs return 404.
Lists use `{count, next, previous, results}` with 25 items per page.

Example client flow (replace the appointment ID with the one returned by the first call):

```sh
curl -X POST http://127.0.0.1:8000/api/appointments/ \
  -H 'Authorization: Token <token>' -H 'Content-Type: application/json' \
  -d '{"Gender":"F","Age":30,"Scholarship":0,"Hipertension":0,"Diabetes":0,"Alcoholism":0,"Handcap":0,"SMS_received":1,"wait_days":2}'
curl -X POST http://127.0.0.1:8000/api/appointments/1/predict/ \
  -H 'Authorization: Token <token>' -H 'Content-Type: application/json' -d '{}'
curl http://127.0.0.1:8000/api/appointments/1/history/ \
  -H 'Authorization: Token <token>'
```

Successful prediction requests return 201 with the saved history record: ID, appointment ID,
requesting user, input snapshot, full probability, binary prediction, threshold, model fingerprint
and timestamp. The decision uses the full probability and the existing threshold. A missing or
unusable artifact returns 503 and creates no history. Appointment creation, admin and history
continue to work without model artifacts.

## Existing trained model

Upstream contains neither the trained model nor the dataset. Copy your **existing trusted**
`best_no_show_model.pkl` into `model/` and, if available, its `threshold.json` into the same folder.
The fallback threshold remains 0.304. You can instead export `MODEL_PATH` and `THRESHOLD_PATH`
as absolute paths. Both Django and FastAPI use the shared `prediction_service` module.
No retraining or changes to the training script are required for this integration. Install the
ML dependency versions used to produce your artifact. Restart workers after replacing files.
If you do not already have artifacts, the original training instructions below explain how to
create them from the separately downloaded dataset.

## Verification

```sh
export DJANGO_DEBUG=true
python manage.py check
python manage.py makemigrations --check --dry-run
python manage.py test appointments -v 2
python -m unittest discover -s tests -v
```

GitHub Actions repeats these checks on pushes and pull requests. Tests use temporary artifacts
and synthetic data and do not require a clinical dataset or a production model.

## Deployment settings

`.env.example` lists settings; export them in your process environment (the file is not loaded
automatically). `DJANGO_DEBUG` defaults to false. Production requires `DJANGO_SECRET_KEY` and
an appropriate `DJANGO_ALLOWED_HOSTS` list. Secure cookies, HTTPS redirection and HSTS are enabled
when debug is off. Run with a production WSGI server, serve collected static files, and configure
a production database and HTTPS termination for your deployment. Do not expose the legacy public
FastAPI demo as the appointment-management API.

---

## Original model training and FastAPI demo

🏥 Patient No-Show Prediction System

A full-stack machine learning application designed to predict whether a patient will miss their scheduled medical appointment. This project features an automated machine learning pipeline, a high-performance REST API, and a lightweight web interface.

🚀 Key Features
1. Automated Model Selection: Evaluates Logistic Regression, Random Forest, XGBoost, and LightGBM, automatically selecting the best performer based on the PR-AUC (Precision-Recall Area Under Curve) metric.

2. Imbalance Handling: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalance in no-show datasets.

3. Targeted Recall Tuning: Selects the decision threshold on validation data for a target recall of 0.75. The final test recall is reported independently and may differ from the validation target.

4. FastAPI Backend: Provides a fast, async-ready RESTful API with automated input validation using Pydantic.

5. Interactive Frontend: A responsive Bootstrap 5 web interface allowing users to input patient details and instantly see the risk assessment.

🛠️ Tech Stack
1. Backend: FastAPI, Uvicorn, Python 3.x

2. Machine Learning: Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, Pandas

3. Frontend: HTML5, Vanilla JavaScript, Bootstrap 5 CSS

📂 Project Structure
1. app.py: The FastAPI application serving the predictions.

2. train_model.py: The machine learning pipeline script used to process data, train models, and export the best model.

3. KaggleV2-May-2016.csv: The medical appointment dataset required for training; download it separately (it is not included in this repository).

4. requirements.txt: Python dependency list.

5. index.html: The frontend user interface.

6. script.js: Client-side logic for form submission and API communication.

7. styles.css: Custom UI styling.

⚙️ Installation & Setup
Run the following commands from the repository root.

1. Install Dependencies: Create a virtual environment and install the required packages:

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```

2. Train the Model: Download `KaggleV2-May-2016.csv` from the [Medical Appointment No Shows dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments) and place it beside `train_model.py`. Then run:

   ```sh
   python train_model.py
   ```

   Alternatively, supply the dataset's location explicitly:

   ```sh
   python train_model.py --data /path/to/KaggleV2-May-2016.csv
   ```

   Training creates `model/best_no_show_model.pkl`, `model/threshold.json`, and `model/evaluation.json` beside the script. The backend uses the model and threshold; the evaluation file records the selected model, training CV scores, split sizes, threshold, and final test metrics. Retrain to replace artifacts generated by the previous training procedure.

   The data is split into stratified training (60%), validation (20%), and test (20%) sets using seed 42. Five-fold cross-validation selects the model using only training data, with preprocessing and SMOTE fitted within each fold. The selected model is fitted on the training set, then validation predictions select the highest threshold meeting the 0.75 recall target. The model and threshold are frozen before the held-out test set is evaluated once for PR-AUC (average precision), precision, and recall. The saved pipeline retains these fitted components without refitting on validation or test rows; SMOTE is used during fitting only.

3. Start the Backend Server: Launch the FastAPI application using Uvicorn

       uvicorn app:app --reload --host 0.0.0.0 --port 8000
   The API will be available at http://localhost:8000.

4. Open the Frontend: Simply open index.html in your preferred web browser. The frontend is configured to communicate directly with the local FastAPI server.

📡 API Reference
Predict No-Show Risk

Evaluates a patient profile and returns the probability of them missing the appointment.

Endpoint: POST /predict

Request Body (JSON):

    {
     "Gender": "F",
     "Age": 30,
     "Scholarship": 0,
     "Hipertension": 0,
     "Diabetes": 0,
     "Alcoholism": 0,
     "Handcap": 0,
     "SMS_received": 1,
     "wait_days": 2
    }

Response (JSON):

    {
     "probability_no_show": 0.412,
     "prediction": 1
    }

  Note: A prediction of 1 indicates a high risk of a no-show, while 0 indicates the patient is likely to attend.


🧪 Training Regression Tests

Run the tests after installing the dependencies:

```sh
python -m unittest discover -s tests -v
```

The tests use synthetic data to check split isolation, validation-only threshold selection, training-only fitting, and prediction consistency after saving the model.
