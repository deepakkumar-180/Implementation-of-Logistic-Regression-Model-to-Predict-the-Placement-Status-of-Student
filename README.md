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
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load CSV file
data = pd.read_csv("placement.csv")

# Remove unnecessary columns
data = data.drop(columns=["sl_no", "salary"])

# Encode categorical columns
le = LabelEncoder()

for column in ["gender", "ssc_b", "hsc_b", "hsc_s",
               "degree_t", "workex", "specialisation"]:
    data[column] = le.fit_transform(data[column])

# Encode target column
data["status"] = data["status"].map({
    "Placed": 1,
    "Not Placed": 0
})

# Separate input and output
X = data.drop("status", axis=1)
y = data["status"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Take user input
print("\nEnter student details:")

gender = input("Gender (M/F): ")
ssc_p = float(input("SSC Percentage: "))
ssc_b = input("SSC Board (Central/Others): ")
hsc_p = float(input("HSC Percentage: "))
hsc_b = input("HSC Board (Central/Others): ")
hsc_s = input("HSC Stream (Commerce/Science/Arts): ")
degree_p = float(input("Degree Percentage: "))
degree_t = input("Degree Type (Sci&Tech/Comm&Mgmt/Others): ")
workex = input("Work Experience (Yes/No): ")
etest_p = float(input("E-Test Percentage: "))
specialisation = input("Specialisation (Mkt&HR/Mkt&Fin): ")
mba_p = float(input("MBA Percentage: "))

# Encode user input using the same category mapping
mappings = {
    "gender": {"F": 0, "M": 1},
    "ssc_b": {"Central": 0, "Others": 1},
    "hsc_b": {"Central": 0, "Others": 1},
    "hsc_s": {"Arts": 0, "Commerce": 1, "Science": 2},
    "degree_t": {"Comm&Mgmt": 0, "Others": 1, "Sci&Tech": 2},
    "workex": {"No": 0, "Yes": 1},
    "specialisation": {"Mkt&Fin": 0, "Mkt&HR": 1}
}

student = [[
    mappings["gender"][gender],
    ssc_p,
    mappings["ssc_b"][ssc_b],
    hsc_p,
    mappings["hsc_b"][hsc_b],
    mappings["hsc_s"][hsc_s],
    degree_p,
    mappings["degree_t"][degree_t],
    mappings["workex"][workex],
    etest_p,
    mappings["specialisation"][specialisation],
    mba_p
]]

# Predict placement
prediction = model.predict(student)

if prediction[0] == 1:
    print("\nPrediction: Placed")
else:
    print("\nPrediction: Not Placed")

```


## Output:

<img width="1488" height="784" alt="image" src="https://github.com/user-attachments/assets/c1914eba-b8b4-40d8-bfc1-833f9902fb15" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
