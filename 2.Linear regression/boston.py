import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model

import matplotlib.pyplot as plt
plt.style.use('ggplot')

boston = datasets.load_boston()
linreg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = modsel.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42
)

linreg.fit(X_train, y_train)

print('mean_squared_error = ' + str(metrics.mean_squared_error(y_train, linreg.predict(X_train))))

y_pred = linreg.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(y_test, linewidth=3, label='ground truth')
plt.plot(y_pred, linewidth=3, label='predicted')
plt.xlabel('test datat points')
plt.ylabel('target value')

plt.figure(2)
plt.plot(y_test, y_pred, 'o')
plt.plot([-10, 60], [-10, 60], 'k--')
plt.axis([-10, 60, -10, 60])
plt.xlabel('ground truth')
plt.ylabel('predicted')

plt.show()