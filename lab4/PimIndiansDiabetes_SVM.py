import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

input_file = 'data/pima-indians-diabetes.csv'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

svc = svm.SVC(kernel='rbf', C=1, gamma=100).fit(X_train, y_train)

y_test_pred = svc.predict(X_test)


cm = confusion_matrix(y_test,y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print(cm)

# dodać metryki jak w decision tree