# Random Forest Classification on Iris Dataset

This project demonstrates the use of a Random Forest classifier on the Iris dataset. The Iris dataset is a classic dataset in the field of machine learning and is frequently used for demonstrating various algorithms.

## Output


https://github.com/sarvesh-2109/Random-Forest/assets/113255836/036b4f8b-41e0-412e-ba98-a1bebfd1ee24




## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
The goal of this project is to classify the species of iris flowers using a Random Forest classifier. The classifier is trained on the Iris dataset and evaluated for its accuracy.

## Dataset
The Iris dataset contains 150 instances of iris flowers, with 50 instances each of three species: Iris setosa, Iris versicolor, and Iris virginica. Each instance has four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- scikit-learn
- seaborn

You can install these libraries using pip:
```bash
pip install pandas scikit-learn seaborn
```

## Usage
1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Train the Random Forest classifier.
4. Evaluate the classifier's accuracy.

Here's a step-by-step breakdown:

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis='columns'), iris.target, test_size=0.2)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Optionally, you can adjust the number of estimators
model = RandomForestClassifier(n_estimators=60)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy with 60 estimators: {accuracy}")
```

## Results
The accuracy of the Random Forest classifier on the test set is printed in the console. You can adjust the number of estimators to see if the model's performance improves.

## License
This project is licensed under the MIT License.
