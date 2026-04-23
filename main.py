# ===============================
# Iris Classification Project
# ===============================

# 1. Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Dataset loaded successfully!")
print("Feature shape:", X.shape)
print("Target classes:", iris.target_names)


# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nData split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Create model
model = DecisionTreeClassifier()

# 5. Train model
model.fit(X_train, y_train)
print("\nModel training completed!")


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Predict new data
print("\n--- Test with user input ---")

while True:
    try:
        print("\nEnter flower features:")

        sepal_length = float(input("Sepal length: "))
        sepal_width = float(input("Sepal width: "))
        petal_length = float(input("Petal length: "))
        petal_width = float(input("Petal width: "))

        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(sample)
        flower_name = iris.target_names[prediction][0]

        print("Predicted class:", flower_name)

    except:
        print("Invalid input!")

    cont = input("Continue? (y/n): ")
    if cont.lower() != "y":
        break