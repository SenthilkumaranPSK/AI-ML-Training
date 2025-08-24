import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Step 1: Create dataset
data = {
    'StudyHours': [2, 5, 8, 3, 6, 1, 4, 7, 9, 2, 5, 7],
    'Attendance': [70, 85, 95, 75, 90, 60, 80, 92, 98, 65, 88, 93],
    'PreviousGrade': ['C', 'B', 'A', 'C', 'B', 'D', 'C', 'A', 'A', 'D', 'B', 'A'],
    'Performance': ['Poor', 'Average', 'Good', 'Poor', 'Good', 'Poor', 'Average', 'Good', 'Good', 'Poor', 'Average', 'Good']
}

df = pd.DataFrame(data)
df.to_csv('student_data.csv', index=False)

# Step 2: Encode grades and performance
grade_encoder = LabelEncoder()
performance_encoder = LabelEncoder()
df['PreviousGrade'] = grade_encoder.fit_transform(df['PreviousGrade'])
df['Performance'] = performance_encoder.fit_transform(df['Performance'])

# Save encoders
pickle.dump(grade_encoder, open('grade_encoder.pkl', 'wb'))
pickle.dump(performance_encoder, open('performance_encoder.pkl', 'wb'))

# Step 3: Split data and train model
X = df[['StudyHours', 'Attendance', 'PreviousGrade']]
y = df['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Save model
pickle.dump(model, open('model_performance.pkl', 'wb'))

print("Model and encoders saved successfully.")
