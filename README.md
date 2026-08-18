# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
# NAME: DEEPAKKUMAR S
# REG NO: 212225230042
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Create the dataset containing student CGPA, aptitude score, and placement status.
2.Split the dataset into training and testing data.
3.Train the Logistic Regression model using the training data.
4.Predict the placement status of students and calculate the model accuracy.
## Program:

```

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create dataset
data = {
    'CGPA': [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 6.8, 7.8, 8.2, 9.2],
    'Aptitude_Score': [55, 60, 65, 70, 75, 85, 58, 68, 72, 90],
    'Placement': [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Input and output
X = df[['CGPA', 'Aptitude_Score']]
y = df['Placement']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display results
print("Actual Values:", y_test.values)
print("Predicted Values:", y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict placement for a new student
cgpa = float(input("Enter CGPA: "))
aptitude = float(input("Enter Aptitude Score: "))

prediction = model.predict([[cgpa, aptitude]])

if prediction[0] == 1:
    print("Student is likely to be PLACED")
else:
    print("Student is likely NOT to be PLACED")

```


## Output:

<img width="1418" height="473" alt="Screenshot 2026-08-18 095818" src="https://github.com/user-attachments/assets/05b57932-8d87-471f-b8b9-966263f15f54" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
