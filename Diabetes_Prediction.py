import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
url ="/content/diabetes (1).csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, names=columns, header=0)
print("Missing values in the dataset:\n")
print(data.isnull().sum())
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.median(), inplace=True)
X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
def predict_diabetes(input_data):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        return "The person is diabetic."
    else:
        return "The person is not diabetic."
def get_user_input():
    print("\nPlease enter the following details for the prediction:")
    pregnancies = float(input("Enter number of Pregnancies: "))
    glucose = float(input("Enter Glucose level: "))
    blood_pressure = float(input("Enter Blood Pressure: "))
    skin_thickness = float(input("Enter Skin Thickness: "))
    insulin = float(input("Enter Insulin level: "))
    bmi = float(input("Enter BMI: "))
    diabetes_pedigree = float(input("Enter Diabetes Pedigree Function: "))
    age = float(input("Enter Age: "))
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    result = predict_diabetes(input_data)
    print("\nPrediction: ", result)
get_user_input()
