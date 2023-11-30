import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
import datetime

ts = datetime.datetime.now()
print(f"start: {ts}")
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

# Create a pipeline with scaling and linear SVM
pipeline = make_pipeline(StandardScaler(), LinearSVC(dual=False, tol=1e-3, random_state=5))

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

ts = datetime.datetime.now()
print(f"stop: {ts}")

# Evaluate the model
print("\nLinearSVC performance on training dataset\n")
print(classification_report(y_train, y_train_pred))
print("\nLinearSVC performance on test dataset\n")
print(classification_report(y_test, y_test_pred))