import numpy as np
import sklearn
import matplotlib.pyplot as plot
from weak_learn import weak_learn

# qustion c
# X_class_0 = np.random.multivariate_normal(mean = [0,0],cov = np.eye(2),size= 500)
# X_class_1 = np.random.multivariate_normal(mean = [2,0],cov= np.eye(2),size=500)
# X = np.vstack((X_class_0,X_class_1))
# y = np.hstack((np.zeros(500,),np.ones(500,)))
# X = np.load('x_random.npy')
# y = np.load('y.npy')
#
# weight_vectors = np.ones(1000,)
# # print(X[:,0])
# # X[:,0] = X[:,0]*10
# # print(X[:,0])
# model = weak_learn()
# model.fit(X, y, gran=1000, weight_vector= weight_vectors)
# np.save('x_random.npy',X)
# np.save('y.npy',y)
# print(model.feature)
# print(model.threshold)

# plot.scatter(X_class_0[:,0],X_class_0[:,1],label = "class_0")
# plot.scatter(X_class_1[:,0],X_class_1[:,1],label = "class_1")
# plot.xlabel("feature1")
# plot.ylabel("feature2")
# plot.xticks([-3,-2,-1,0,1,2,3,4,5])
# plot.yticks([-3,-2,-1,0,1,2,3])
# plot.legend()
# plot.show()

# question d:
f = open('fashion57_train.txt')
lines = f.readlines()
X_train = []
y_train = np.hstack((np.zeros(32), np.ones(28)))

for i in lines:

    linestr = i.strip()
    linestrlist = linestr.split(',')
    results = list(map(float, linestrlist))
    X_train.append(results)

X_train = np.array(X_train)


f = open('fashion57_test.txt')
lines = f.readlines()
X_test = []
y_test = np.hstack((np.zeros(195), np.ones(205)))

for i in lines:

    linestr = i.strip()
    linestrlist = linestr.split(',')
    results = list(map(float, linestrlist))
    X_test.append(results)

X_test = np.array(X_test)

weight_vectors = np.ones(60,)
model = weak_learn()
model.fit(X_train, y_train, gran =60, weight_vector= weight_vectors)
print("apparent_error", model.evaluate(X_train,y_train))
print("test_error", model.evaluate(X_test, y_test))