import random
import matplotlib.pyplot as plt
import time

def calcSum(x):
    sum = 0
    for i in range(0, len(x)):
        sum += x[i]

    return sum

#y = 5 + 2 * x1

x_train = range(0, 200)
x_test = range(200, 250)
y_train = []
y_test = []

f = open("./trained_data.csv", 'w', encoding = 'utf-8')
for i in range(0, 200):
    data_line = ""
    y = 5 + 2 * i + 0.5 * random.random()
    y_train.append(y)
    data_line += str(y)

    data_line += ','
    data_line += str(i)
    
    data_line += "\n"

    f.write(data_line)
f.close()

f = open("./test_data.csv", 'w', encoding = 'utf-8')
for i in range(200, 250):
    data_line = ""
    y = 5 + 2 * i + 0.5 * random.random()
    y_test.append(y)
    data_line += str(y)

    data_line += ','
    data_line += str(i)
    
    data_line += "\n"

    f.write(data_line)
f.close()

theta0 = 0
theta1 = 0
alpha = 0.00001

sumy = calcSum(y_train)
sumx = calcSum(x_train)

theta1 = sumy / sumx
theta0 = y_train[0] - theta1 * x_train[0]
print('theta1 = ' + str(theta1) + ' theta0 = ' + str(theta0))

while True:
    tmp1 = 0
    tmp0 = 0
    for i in range(0, 200):
        tmp1 = tmp1 + (y_train[i] - (theta1 * x_train[i] + theta0)) * x_train[i]
        tmp0 = tmp0 + (y_train[i] - (theta1 * x_train[i] + theta0))

    old_theta1 = theta1
    old_theta0 = theta0
    theta1 = theta1 + alpha * tmp1 / 200
    theta0 = theta0 + alpha * tmp0 / 200
    tmp1 = 0
    tmp0 = 0

    print('theta1 = ' + str(theta1) + ' theta0 = ' + str(theta0))

    e = ((old_theta1 - theta1) * (old_theta1 - theta1) + (old_theta0 - theta0) * (old_theta0 - theta0))
    if e < 0.000003:
        break

    # time.sleep(1)

    
print('y = ' + str(theta1) + ' * x + ' + str(theta0))

y_pred = []
for i in range(200, 250):
    y_pred.append(theta1 * x_test[i - 200] + theta0)

plt.plot(y_test, 'b.')
plt.plot(y_pred, 'r.')
plt.show()