# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the dataset to initiate the analysis.

2.Examine the dataset to identify patterns, distributions, and relationships.

3.Determine the most important features to enhance model accuracy and efficiency.

4.Separate the dataset into training and testing sets for effective validation.

5.Use the training data to build and train the model.

6.Measure the model’s performance on the test data with relevant metrics.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: SUBASHRAM T
RegisterNumber: 212225040430
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
print(data.head())
print(data.columns)
X = data.drop(columns=['Class']) 
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("="*40)
print("Name :SUBASHRAM T")
print("Reg. No :212225040430")
print("="*40)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show() 
*/
```

## Output:
<img width="1243" height="705" alt="Screenshot 2026-03-25 141230" src="https://github.com/user-attachments/assets/b1e04e37-195b-4585-aeba-010aa843ffa4" />
<img width="1231" height="316" alt="Screenshot 2026-03-25 141247" src="https://github.com/user-attachments/assets/d8f9e8c8-59cd-4c86-bc60-71e22f94cb01" />
<img width="1207" height="572" alt="Screenshot 2026-03-25 141304" src="https://github.com/user-attachments/assets/9e49a6e1-c93d-4eba-9cf4-2e831a033a50" />
## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
