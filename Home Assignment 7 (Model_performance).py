import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Load dataset
df = pd.read_csv('student_data.csv')

# Step 2: Encode categorical columns
grade_encoder = LabelEncoder()
performance_encoder = LabelEncoder()
df['PreviousGrade'] = grade_encoder.fit_transform(df['PreviousGrade'])  # A=0, B=1, C=2...
df['Performance'] = performance_encoder.fit_transform(df['Performance'])  # Poor=0, Average=1, Good=2

# Save encoders for inference
with open('grade_encoder.pkl', 'wb') as f:
    pickle.dump(grade_encoder, f)
with open('performance_encoder.pkl', 'wb') as f:
    pickle.dump(performance_encoder, f)

# Step 3: Features and target
X = df[['StudyHours', 'Attendance', 'PreviousGrade']]
y = df['Performance']

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Save the model
with open('model_performance.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as model_performance.pkl")
