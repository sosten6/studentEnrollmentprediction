import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('student_data.csv')

# Display the first few rows of the dataset
print("Initial Data Sample:")
print(df.head(), "\n")

# Data Cleaning
# Fill missing values for numeric columns with the mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables into numeric format
df = pd.get_dummies(df, drop_first=True)

# Display the data types of the DataFrame
print("Data Types After Encoding:")
print(df.dtypes, "\n")

# Define features and target variables
X = df.drop(columns=['enrollment_status', 'graduation_status'])  # Features
y_enrollment = df['enrollment_status']  # Target for enrollment prediction
y_graduation = df['graduation_status']  # Target for graduation prediction

# Split the data into training and testing sets for enrollment prediction
X_train, X_test, y_train_enrollment, y_test_enrollment = train_test_split(X, y_enrollment, test_size=0.2, random_state=42)

# Model for Enrollment Prediction
enrollment_model = RandomForestClassifier(n_estimators=100, random_state=42)
enrollment_model.fit(X_train, y_train_enrollment)

# Predicting Enrollment
y_pred_enrollment = enrollment_model.predict(X_test)

# Split the data into training and testing sets for graduation prediction
X_train_grad, X_test_grad, y_train_grad, y_test_grad = train_test_split(X, y_graduation, test_size=0.2, random_state=42)

# Model for Graduation Support Prediction
graduation_model = RandomForestClassifier(n_estimators=100, random_state=42)
graduation_model.fit(X_train_grad, y_train_grad)

# Predicting Graduation
y_pred_grad = graduation_model.predict(X_test_grad)

# Create a figure for all outputs
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Enrollment Prediction Report
enrollment_report = classification_report(y_test_enrollment, y_pred_enrollment, output_dict=True)
sns.heatmap(confusion_matrix(y_test_enrollment, y_pred_enrollment), annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('Enrollment Confusion Matrix')
axes[0, 0].set_xlabel('Predicted')
axes[0, 0].set_ylabel('Actual')

# Graduation Prediction Report
graduation_report = classification_report(y_test_grad, y_pred_grad, output_dict=True)
sns.heatmap(confusion_matrix(y_test_grad, y_pred_grad), annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('Graduation Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# Displaying the classification reports as text in the lower part of the figure
axes[1, 0].axis('off')  # Turn off the axis
axes[1, 0].text(0.5, 0.5, 'Enrollment Prediction Report:\n' + classification_report(y_test_enrollment, y_pred_enrollment),
                 fontsize=12, ha='center', va='center', wrap=True)

axes[1, 1].axis('off')  # Turn off the axis
axes[1, 1].text(0.5, 0.5, 'Graduation Prediction Report:\n' + classification_report(y_test_grad, y_pred_grad),
                 fontsize=12, ha='center', va='center', wrap=True)

plt.tight_layout()  # Adjust the layout
plt.show()  # Show all plots at once

# Example Usage Function for Predictions
def predict_enrollment(features):
    """Predict enrollment status based on provided features."""
    return enrollment_model.predict([features])

def predict_graduation(features):
    """Predict graduation success based on provided features."""
    return graduation_model.predict([features])

# Example input for predictions
example_features = [3.5, 30, 1, 0]  # Adjust based on your dataset features
print("Example Predictions:")
print('Predicted Enrollment Status:', predict_enrollment(example_features)[0])
print('Predicted Graduation Status:', predict_graduation(example_features)[0])