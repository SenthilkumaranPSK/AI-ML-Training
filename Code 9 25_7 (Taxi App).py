from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('taxi_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    population = float(request.form['population'])
    income = float(request.form['income'])
    parking = float(request.form['parking'])
    riders = float(request.form['riders'])

    # Predict using model
    prediction = model.predict([[population, income, parking, riders]])[0]

    # Render result
    return render_template('result.html', 
                           population=population,
                           income=income,
                           parking=parking,
                           riders=riders,
                           prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
