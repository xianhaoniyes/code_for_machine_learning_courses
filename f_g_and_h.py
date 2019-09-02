import numpy as np
import sklearn
from adaboost_implementation import adaboost
import math
import matplotlib.pyplot as plot


#question f

X = np.load('x_random.npy')
y = np.load('y.npy')
model = adaboost()
weights = model.fit(X, y, gran = 100, T=100)
print(model.evaluate(X,y))

index = []
for i in range(0,1000):
    index.append(i)

def key(item):

    return weights[item]

index.sort(key=key)

small = index[0:200]
large = index[-200:]
st = [1,0]
print(weights[small[:]])
print(weights[large[:]])
print(small)
print(large)
dist_small = []
dist_large = []

print("samll")
for item in small:
    point  = np.array([X[item][0],X[item][1]])
    dist_small.append(np.linalg.norm(point - st))

print(np.mean(dist_small))


print("large")

for item in large:
    point  = np.array([X[item][0],X[item][1]])
    dist_large.append(np.linalg.norm(point - st))

print(np.mean(dist_large))

print("no")

x_small = X[small[:]]
print(np.shape(x_small))

x_large = X[large[:]]

plot.scatter(x_small[:,0],x_small[:,1],label = "points_small_weight")
plot.scatter(x_large[:,0],x_large[:,1],label = "points_large_weight")

plot.xlabel("feature1")
plot.ylabel("feature2")
plot.xticks([-3,-2,-1,0,1,2,3,4,5])
plot.yticks([-3,-2,-1,0,1,2,3])
plot.legend()
plot.show()





# f = open('fashion57_train.txt')
# lines = f.readlines()
# X_train = []
# y_train = np.hstack((np.zeros(32), np.ones(28)))
#
# for i in lines:
#
#     linestr = i.strip()
#     linestrlist = linestr.split(',')
#     results = list(map(float, linestrlist))
#     X_train.append(results)
#
# X_train = np.array(X_train)
#
# f = open('fashion57_test.txt')
# lines = f.readlines()
# X_test = []
# y_test = np.hstack((np.zeros(195), np.ones(205)))
#
# for i in lines:
#
#     linestr = i.strip()
#     linestrlist = linestr.split(',')
#     results = list(map(float, linestrlist))
#     X_test.append(results)

# X_test = np.array(X_test)
# question g1
# error_results=[]
# for t in range (10,120,10):
#     model = adaboost()
#     weights = model.fit(X_train,y_train,gran = 10,T=t)
#     a =  model.evaluate(X_test,y_test)
#     error_results.append(a)
#
# np.save('error_results.npy',error_results)
# print(error_results)

# error_reuslt = np.load("error_results.npy")
# print(error_reuslt)
# error_reuslt = error_reuslt[0:-1]
# xi = [10,20,30,40,50,60,70,80,90,100]
# plot.plot(xi,error_reuslt, label = 'test_error')
# plot.ylabel('error')
# plot.xlabel('number of iterations')
#
#
# plot.legend()
# plot.show()
#question g2

# model = adaboost()
# weights = model.fit(X_train,y_train,gran = 10,T=60)
# a =  model.evaluate(X_test,y_test)
# print(a)
# index = []
# for i in range(0, 60):
#     index.append(i)
#
# def key(item):
#
#     return weights[item]
#
# index.sort(key=key, reverse=True)
#
# print(index[:])
# print(weights[index[:]])



#question h
# X_train = np.vstack((X_train[6:12], X_train[40:46]))
#
# y_train = np.hstack((np.zeros(6), np.ones(6)))
# model = adaboost()
# weights = model.fit(X_train,y_train,gran = 10,T=60)
# a =  model.evaluate(X_test,y_test)
# print(a)

# x = [2,4,6,10,15,20]
#
# y = [0.543,0.524,0.510,0.490,0.372,0.30]
#
# plot.plot(x,y, label = 'test_error')
# plot.xticks([2,4,6,10,15,20])
# plot.ylabel('error')
# plot.xlabel('training objects per class')
# plot.legend()
# plot.show()