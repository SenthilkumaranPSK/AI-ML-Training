#Train and Save Model
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'text': [
        'Win money now',
        'Limited offer just for you',
        'Hi, how are you?',
        'Call me tomorrow',
        'Free tickets available',
        'Congratulations, you won a prize!',
        'Are you coming to the party?',
        'Lets grab lunch today',
        'Earn extra cash fast',
        'Meeting at 3 PM',
        'Click here to claim your reward',
        'Reminder: Doctor appointment at 5 PM',
        'Your loan is approved',
        'Can we talk later tonight?',
        'Exclusive deal just for you',
        'Don’t forget the groceries',
        'Act now to get 50% off',
        'Dinner at 8?',
        'Free gift card inside',
        'Happy birthday! 🎉',
        'You have been selected!',
        'Let’s catch up this weekend',
        'Hot singles in your area',
        'Submit your assignment by 11 PM',
        'You are a winner!'
    ],
    'Label': [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("\nModel and vectorizer saved!")
