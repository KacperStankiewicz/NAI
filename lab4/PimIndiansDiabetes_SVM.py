import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Load the dataset
input_file = 'data/pima-indians-diabetes.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# split data into training and test in 1:4 ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5)

# standarize data
scaler = StandardScaler()
X_train_standarized = scaler.fit_transform(X_train)
X_test_standarized = scaler.transform(X_test)

# search for best parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_standarized, y_train)
best_params = grid_search.best_params_

# create classificator with best parameters
svc = svm.SVC(**best_params).fit(X_train_standarized, y_train)

# make predictions
y_train_pred = svc.predict(X_train)
y_test_pred = svc.predict(X_test_standarized)

# create and print confussion matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print(cm)
plt.show()

# print report
print("\nSVC performance on training dataset\n")
print(classification_report(y_train, y_train_pred, zero_division=0.0))
print("\nSVC performance on test dataset\n")
print(classification_report(y_test, y_test_pred, zero_division=0.0))
