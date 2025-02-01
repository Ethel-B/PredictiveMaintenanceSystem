#An AI maintenance system for Dawn Industries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Loading equipment data
data = pd.read_csv('equipment_data.csv')


# Preprocessing data
X = data['target']
y = data['target']


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# Scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_train_scaled= scaler.fit_transform(y_train)
y_test_scaled= scaler.fit_transform(y_test)


# Train Isolation Forest model
model = IsolationForest(n_estimators=100, random_state=40)
model.fit(X_train_scaled)
model.fit(y_train_scaled)

# Predicting anomalies
y_pred = model.predict(X_test_scaled)

#Time series forcasting
future_values = model.predict(X_test_scaled)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))


# Plotting anomalies
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred)
plt.plot(future_values)
plt.show()
plt.show()


