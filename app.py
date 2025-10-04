from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_form', methods=['POST'])
def predict_form():

    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DPF = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])


    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    prediction = model.predict(features)[0]


    result = "Likely to have diabetes" if prediction == 1 else "Unlikely to have diabetes"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
