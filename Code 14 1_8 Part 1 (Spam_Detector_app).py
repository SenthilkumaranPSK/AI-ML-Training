from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
with open('model.pkl', 'rb') as model_file:
 model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vec_file:
 vectorizer = pickle.load(vec_file)

@app.route('/', methods=['GET', 'POST'])
def index():
  prediction = None
  if request.method == 'POST':
   message = request.form['message']
   message_vector = vectorizer.transform([message])
   pred = model.predict(message_vector)[0] 
   prediction = "Spam" if pred == 1 else "Not Spam"
  return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
  app.run(debug=True)
