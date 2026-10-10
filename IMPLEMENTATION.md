# Django implementation steps

1. [x] Fetch and inspect the existing FastAPI and training code.
2. [x] Extract inference into `prediction_service`: lazy loading, script-relative artifact paths,
   optional environment overrides, original 0.304 fallback and `probability >= threshold` rule.
   Keep the fitted model and training script unchanged. FastAPI also calls this service.
3. [x] Add the Django project and SQLite development database, appointment and prediction-history
   models, validators, database constraints, and a committed initial migration.
4. [x] Register both models in admin. Saved prediction fields are read-only and cannot be added
   or deleted through admin, preserving the model result for inspection.
5. [x] Add DRF appointment creation/list/detail, saved-appointment prediction, per-appointment
   history and global history list/detail. Save exact inputs, raw probability, threshold,
   model SHA-256 fingerprint, requesting user and creation time for each prediction.
6. [x] Require token or session authentication. Assign owners on the server, reject owner
   overrides, scope lists and details to the owner, and allow staff to review all records.
   History is read-only through the API; appointment updates/deletions are intentionally
   absent from this initial API. Staff can edit appointments through Django admin.
7. [x] Test numeric and categorical validation, missing fields, authentication, ownership,
   staff access, persistence, history snapshots, exact threshold decisions, shared API
   prediction consistency, corrupted/missing artifacts, and database constraints.
8. [x] Document local setup, credentials, endpoints, artifacts, and deployment settings;
   add GitHub Actions to run Django and existing training tests.

## Design choices

- Use the original model input names, including `Hipertension` and `Handcap`, so no retraining
  or transformation of the saved pipeline is required.
- `wait_days` remains supplied explicitly with the original -7..365 range. This first version
  does not invent scheduling dates, patient identifiers, or patient registration requirements.
- Predict only saved appointments, using `POST /api/appointments/{id}/predict/` with `{}`.
  Input overrides are rejected. Every successful call creates a distinct history row.
- Django history returns full probability precision. The legacy FastAPI response still rounds
  to three decimals; both APIs classify using the unrounded value.
- Inference artifacts load once per process; restart workers after replacing artifacts.
  Fingerprints and saved thresholds allow old predictions to remain interpretable.
- No production model or dataset is committed upstream. Test doubles and temporary synthetic
  models verify behavior; they are never installed as production artifacts.
- The static frontend remains a client of the original FastAPI demo. Use Django's browsable
  API with session login, or an API client with an issued token, for managed appointments.
- Production requires a secret, explicit host configuration, HTTPS and a deployment database
  suited to the workload. SQLite is the local starting point. Use the same ML library versions
  used to serialize your existing trusted model; joblib artifacts are not portable across all
  versions and must not come from untrusted uploads.

## Local verification results

- 21 Django/API/service tests passed, including a temporary fitted scikit-learn pipeline
  compared with direct inference, session CSRF enforcement and malformed JSON body validation.
- All 5 existing training regression tests passed.
- Django system checks passed; no migration drift detected.
- Initial migrations applied successfully to the local SQLite database.
- `git diff --check` passed; `train_model.py` has no diff.

Production artifact parity cannot be checked until the original trained artifact is supplied.
The changes are maintained on `codex/django-appointment-management`. No deployment was performed.
