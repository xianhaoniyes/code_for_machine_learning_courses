from __future__ import division
import numpy as np
import csv


datafile = open('labels _back.csv', 'r')
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



a = real_data[:]
k = a[a == 1]
print(len(k))

