'''
Authors:
Rutkowski Marcin s12497
Stankiewicz Kacper s22619

Environment preparation:
pip install numpy
pip install matplotlib
pip install sklearn
pip install seaborn
pip install pandas
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
data = pd.read_csv('data/airlines_delay.csv')

# Preprocess the data: Convert categorical variables to numerical
encoder = OneHotEncoder(sparse_output=False)
categorical_columns = ['Airline', 'AirportFrom', 'AirportTo']  # Add other categorical columns if necessary
encoded_cats = encoder.fit_transform(data[categorical_columns])

# Create a new DataFrame with the encoded variables
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))

# Drop the original categorical columns and concatenate the encoded ones
data = data.drop(categorical_columns, axis=1)
data = pd.concat([data, encoded_df], axis=1)

# Split the data
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Train the model
classifier = DecisionTreeClassifier(random_state=5)
classifier.fit(X_train, y_train)

# Predictions
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

# Evaluate the model
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, y_train_pred))
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
