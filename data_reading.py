import numpy as np
f = open('twoGaussians.txt')
lines = f.readlines()
X = []
y = []
for i in lines:

    linestr = i.strip()
    linestrlist = linestr.split(',')
    results = list(map(float, linestrlist))
    X.append(results[:-1])
    y.append(int(results[-1]))


X = np.array(X)
y = np.array(y)

np.save('vectors.npy', X)
np.save('labels.npy', y)