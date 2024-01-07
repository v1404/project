from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\project wok\\df111.csv")

# Separate features and target variables
X = data.drop(['STAGE', 'SURGERY', 'CLASS'], axis=1)
y_stage = data['STAGE']
y_surgery = data['SURGERY']
y_class = data['CLASS']

# Encode categorical columns using Label Encoding
label_encoder = LabelEncoder()
categorical_cols = ['SEX', 'TOPOGRAPHY', 'T', 'N', 'M']
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Split dataset into training and testing
X_train, X_test, y_stage_train, y_stage_test, y_surgery_train, y_surgery_test, y_class_train, y_class_test = train_test_split(
    X, y_stage, y_surgery, y_class, test_size=0.2, random_state=42)

# Create Gaussian Naive Bayes models for each target
model_stage = GaussianNB()
model_surgery = GaussianNB()
model_class = GaussianNB()

# Train the models
model_stage.fit(X_train, y_stage_train)
model_surgery.fit(X_train, y_surgery_train)
model_class.fit(X_train, y_class_train)

# Try using StandardScaler to standardize numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

num_cols = ['AGE']
X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test_scaled[num_cols])

# Retrain the models with scaled features
model_stage.fit(X_train_scaled, y_stage_train)
model_surgery.fit(X_train_scaled, y_surgery_train)
model_class.fit(X_train_scaled, y_class_train)

# Define a route to render the input form
@app.route('/')
def index():
    return render_template('index.html', X=X)

# Define a route to handle the form submission and display predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        patient_data = {}
        for col in X.columns:
            if col in categorical_cols:
                le = LabelEncoder()
                patient_data[col] = le.fit_transform([request.form[col]])[0]
            else:
                patient_data[col] = float(request.form[col])

        # Validate user input
        if any(val < 0 for val in patient_data.values()):
            return "Invalid input. Please enter non-negative values."
        else:
            # Use the models to predict the risk of the patient
            features_array = np.array(list(patient_data.values())).reshape(1, -1)

            stage_prediction = model_stage.predict(features_array)[0]
            surgery_prediction_number = int(model_surgery.predict(features_array)[0])
            
            surgery_names = {
                1: 'Hartmann rectosigmoidian resection',
                2: 'Dixon rectosigmoidian resection',
                3: 'Right hemicolectomy',
                4: 'Left hemicolectomy',
                5: 'Segmental colectomy',
                6: 'Rectal biopsy',
                7: 'Ileotransverse-anastomosis',
                8: 'Abdominoperineal rectal amputation',
                9: 'Left iliac colostomy'
            }

            surgery_prediction_name = f"{surgery_prediction_number}: {surgery_names.get(surgery_prediction_number, 'Unknown')}"

            class_prediction = model_class.predict(features_array)[0]

            return render_template('result.html', stage_prediction=stage_prediction,
                                   surgery_prediction_number=surgery_prediction_number,
                                   surgery_prediction_name=surgery_prediction_name,
                                   class_prediction=class_prediction)
