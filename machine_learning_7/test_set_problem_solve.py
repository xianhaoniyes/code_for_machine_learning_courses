import numpy as np
import csv
from sklearn.ensemble import GradientBoostingClassifier

# train ###########################################################################

datafile = open('train.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
data = []
for row in datareader:
    data.append(row)

real_data = []
for row in data:
    real_data_row = []
    for item in row:
        real_data_row.append(int(float(item)))
    real_data.append(real_data_row)

real_data = np.array(real_data)

real_data[:, 12] = real_data[:, 12] - 1
real_data[:, 14] = real_data[:, 14] - 1


data = real_data[:, 0:13]
labels = real_data[:, 14]

boost = GradientBoostingClassifier(n_estimators=300)

boost.fit(data, labels)

datafile.close()
# test ################################################################################################

datafile = open('test.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
data = []
for row in datareader:
    data.append(row)

test_data = []
for row in data:
    test_data_row = []
    for item in row:
        test_data_row.append(int(float(item)))
    test_data.append(test_data_row)

test_data = np.array(test_data)
print(np.shape(test_data))

test_data[:, 12] = test_data[:, 12] - 1
feature_13 = test_data[:, 12]

pre = boost.predict_proba(test_data)[:, 1]

pre[list(filter(lambda i : feature_13[i] == 0 and pre[i] >= 0.6, range(0,len(pre))))] = 1
pre[list(filter(lambda i : feature_13[i] == 0 and pre[i] < 0.6, range(0,len(pre))))] = 0

pre[list(filter(lambda i : feature_13[i] == 1 and pre[i] >= 0.7, range(0,len(pre))))] = 1
pre[list(filter(lambda i : feature_13[i] == 1 and pre[i] < 0.7, range(0,len(pre))))] = 0


pre = pre + 1

pre = np.array(pre, dtype=int)

#some test /////////////////////////////////////////////////////////
a = pre[:]
k = a[a == 1]
print(len(k))
pre =  np.array(map(str, pre))
#/////////////////////////////////////////////////////////


pre = pre.tolist()


with open('test_labels.csv', 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(pre)

