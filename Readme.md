🏥 Patient No-Show Prediction System

A full-stack machine learning application designed to predict whether a patient will miss their scheduled medical appointment. This project features an automated machine learning pipeline, a high-performance REST API, and a lightweight web interface.

🚀 Key Features
1. Automated Model Selection: Evaluates Logistic Regression, Random Forest, XGBoost, and LightGBM, automatically selecting the best performer based on the PR-AUC (Precision-Recall Area Under Curve) metric.

2. Imbalance Handling: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalance in no-show datasets.

3. Targeted Recall Tuning: Calibrates the decision threshold specifically to achieve a target recall of 0.75, ensuring high detection rates for potential no-shows.

4. FastAPI Backend: Provides a fast, async-ready RESTful API with automated input validation using Pydantic.

5. Interactive Frontend: A responsive Bootstrap 5 web interface allowing users to input patient details and instantly see the risk assessment.

🛠️ Tech Stack
1. Backend: FastAPI, Uvicorn, Python 3.x

2. Machine Learning: Scikit-Learn, XGBoost, LightGBM, Imbalanced-Learn, Pandas

3. Frontend: HTML5, Vanilla JavaScript, Bootstrap 5 CSS

📂 Project Structure
1. app.py: The FastAPI application serving the predictions.

2. train.py: The machine learning pipeline script used to process data, train models, and export the best model.

3. KaggleV2-May-2016.csv: The core medical appointment dataset required for training the model.

4. requirements.txt: Python dependency list.

5. index.html: The frontend user interface.

6. script.js: Client-side logic for form submission and API communication.

7. styles.css: Custom UI styling.

⚙️ Installation & Setup
1. Install Dependencies: Ensure you have Python installed. Install the required packages via pip:

       pip install fastapi==0.111.0 uvicorn[standard]==0.29.0 pydantic pandas numpy scikit-learn imbalanced-learn lightgbm joblib xgboost

2. Train the Model: The backend requires a trained model to function. Make sure the dataset KaggleV2-May-2016.csv is in the correct directory. Run the training script to generate the model artifacts:

        python train.py
This will create a model/ directory containing best_no_show_model.pkl and threshold.json.

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

