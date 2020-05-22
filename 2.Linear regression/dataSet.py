import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection as modsel
from sklearn import linear_model

boston = datasets.load_boston()
linreg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = modsel.train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=42
)

f = open("./trained_data.csv", 'w', encoding = 'utf-8')
for i in range(0, len(X_train)):
    data_line = ""
    data_line += str(y_train[i])

    for j in range(0, 13):
        data_line += ','
        data_line += str(X_train[i, j])
    
    data_line += "\n"

    f.write(data_line)
f.close()

f = open("./test_data.csv", 'w', encoding = 'utf-8')
for i in range(0, len(X_test)):
    data_line = ""
    data_line += str(y_test[i])

    for j in range(0, 13):
        data_line += ','
        data_line += str(X_test[i, j])
    
    data_line += "\n"

    f.write(data_line)
f.close()