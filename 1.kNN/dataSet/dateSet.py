#python 3.7
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data_count = 1000

data=make_blobs(n_samples = data_count, centers = 2, center_box = (0, 10), random_state = 2)
X,y=data
#print(X)
plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.spring,edgecolor='k')
plt.show()

f = open("./trained_data.csv", 'w', encoding = 'utf-8')

for index in range(0, (int)(data_count*0.8)):
    data_line = ""
    if(y[index] == 0):
        data_line = "A"
    elif(y[index] == 1):
        data_line = "B"

    data_line += ","
    data_line += str(X[index, 0])
    data_line += ","
    data_line += str(X[index, 1])
    data_line += "\n"

    f.write(data_line)

f.close()

f = open("./test_data.csv", 'w', encoding = 'utf-8')

for index in range((int)(data_count*0.8), data_count):
    data_line = ""
    if(y[index] == 0):
        data_line = "A"
    elif(y[index] == 1):
        data_line = "B"

    data_line += ","
    data_line += str(X[index, 0])
    data_line += ","
    data_line += str(X[index, 1])
    data_line += "\n"

    f.write(data_line)

f.close()