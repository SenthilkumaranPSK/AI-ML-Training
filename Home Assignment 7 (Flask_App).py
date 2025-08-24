from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model_performance.pkl', 'rb'))
grade_encoder = pickle.load(open('grade_encoder.pkl', 'rb'))
performance_encoder = pickle.load(open('performance_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance'])
        grade_input = request.form['previous_grade'].strip().upper()

        grade_numeric = grade_encoder.transform([grade_input])[0]
        input_data = np.array([[study_hours, attendance, grade_numeric]])
        pred_encoded = model.predict(input_data)[0]
        prediction = performance_encoder.inverse_transform([pred_encoded])[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
