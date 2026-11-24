import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# 1. Dataset load karo
df = pd.read_csv('student_data.csv')

# 2. Input (X) aur Output (y) alag karo
X = df[['Study_Hours', 'Attendance', 'Previous_Score']]
y = df['Final_Score']

# 3. 80% Training aur 20% Testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model train karo
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Accuracy check karo
predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)
print("Model Accuracy (R2 Score):", round(accuracy * 100, 2), "%")

# 6. Trained model save karo
joblib.dump(model, 'score_model.pkl')
print("Model saved successfully as score_model.pkl!")