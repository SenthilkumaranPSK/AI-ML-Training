import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load CSV file
df = pd.read_csv('taxi.csv')

# Define input features (X) and target variable (y)
X = df[['Population', 'Monthlyincome', 'Averageparkingpermonth', 'Numberofweeklyriders']]
y = df['Priceperweek']

# Split the dataset (optional, for model evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'taxi_model.pkl')
print("✅ Model trained and saved as 'taxi_model.pkl'")
