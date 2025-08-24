from flask import Flask, request, render_template, flash
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
model = pickle.load(open(MODEL_PATH, 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        val1 = float(request.form['feature1'])
        val2 = float(request.form['feature2'])
        val3 = float(request.form['feature3'])
        val4 = float(request.form['feature4'])

        input_data = [[val1, val2], [val3, val4]]
        predictions = model.predict(input_data)

        return render_template('result.html', prediction=predictions)

    except ValueError:
        flash("Invalid input! Please enter valid numbers only.")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
