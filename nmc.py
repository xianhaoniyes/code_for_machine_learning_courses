import numpy as np
import statistics


class nearstMeanClassifer:

    def __init__(self, dimension,learning_rate, lamda):
        self.m_nega = np.zeros(shape = (1,dimension))
        self.m_posi = np.zeros(shape =(1,dimension))
        self.a = 0.0
        self.learning_rate = learning_rate
        self.lamda =lamda
        self.dimension = dimension

    def fit(self, X, y):
        learning_rate = self.learning_rate
        X_posi = []
        X_nega = []
        for i in range (0,len(y)):
            if y[i] ==1:
                X_posi.append(X[i])
            else:
                X_nega.append(X[i])

        X_posi = np.array(X_posi)
        X_nega = np.array(X_nega)

        for i in range (0,1000):
            dm_posi, dm_nega, dm_a = self.devariate(X_posi, X_nega)
            self.m_posi = self.m_posi - learning_rate*dm_posi
            self.m_nega = self.m_nega - learning_rate*dm_nega
            self.a = self.a -learning_rate*dm_a
            print(i)
            print(self.m_posi)   #ok

        learning_rate = learning_rate/10

        for i in range (0,2000):
            dm_posi, dm_nega, dm_a = self.devariate(X_posi, X_nega)
            self.m_posi = self.m_posi - learning_rate*dm_posi
            self.m_nega = self.m_nega - learning_rate*dm_nega
            self.a = self.a -learning_rate*dm_a
            print(i)
            print(self.m_posi) #ok

        learning_rate = learning_rate / 2

        for i in range (0,5000):
            dm_posi, dm_nega, dm_a = self.devariate(X_posi, X_nega)
            self.m_posi = self.m_posi - learning_rate*dm_posi
            self.m_nega = self.m_nega - learning_rate*dm_nega
            self.a = self.a -learning_rate*dm_a
            print(i)
            print(self.m_posi)  #ok
        #
        learning_rate = learning_rate / 2

        for i in range (0,30000):
            dm_posi, dm_nega, dm_a = self.devariate(X_posi, X_nega)
            self.m_posi = self.m_posi - learning_rate*dm_posi
            self.m_nega = self.m_nega - learning_rate*dm_nega
            self.a = self.a -learning_rate*dm_a
            print(i)
            print(self.m_posi)  # ok

        learning_rate = learning_rate / 5

        for i in range (0, 30000):
            dm_posi, dm_nega, dm_a = self.devariate(X_posi, X_nega)
            self.m_posi = self.m_posi - learning_rate*dm_posi
            self.m_nega = self.m_nega - learning_rate*dm_nega
            self.a = self.a -learning_rate*dm_a
            print(i)
            print(self.m_posi)

    def devariate(self,X_posi,  X_nega):

        dm_posi = np.sum(np.sign(self.m_posi-X_posi), axis=0) + self.lamda*(np.sign(self.m_posi-self.m_nega + self.a))
        dm_nega = np.sum(np.sign(self.m_nega-X_nega), axis=0) - self.lamda*(np.sign(self.m_posi-self.m_nega + self.a))
        dm_a = self.lamda*self.devariate_a()
        return dm_posi, dm_nega, dm_a

    def devariate_a(self):

        current =self.m_posi-self.m_nega
        current = np.squeeze(current)
        current.sort()
        # print(current)
        i = 0
        for j in range(0, self.dimension):
            if self.a >= current[j]:
                i = i+1
            else:
                break

        return -self.dimension + 2*i


    def predict(self, X):
        result = []
        for i in X:
            pred = (np.sum(np.abs(i-self.m_posi))<np.sum(np.abs(i-self.m_nega)))
            if pred:
                result.append(1)
            else:
                result.append(0)
        return np.array(result)

    def evaluate(self, X_test, y_test_true, X_train, y_train_true):
        y_test_pred = self.predict(X_test)
        true_error = 0
        for i in range(0, len(y_test_true)):
            if y_test_pred[i] !=  y_test_true[i]:
                true_error = true_error+1

        y_train_pred = self.predict(X_train)

        apparent_error = 0
        for i in range(0,len(y_train_true)):
            if y_train_pred[i] != y_train_true[i]:
                apparent_error = apparent_error +1

        apparent_error = apparent_error/len(y_train_true)
        true_error = true_error/len(y_test_true)
        return apparent_error, true_error

f = open('digits.txt')
lines = f.readlines()
X = []
y = []

for i in lines:

    linestr = i.strip()
    linestrlist = linestr.split(',')
    results = list(map(float, linestrlist))
    X.append(results)


for i in range (0,10000):
    y.append(1)
for i in range (0,10000):
    y.append(0)

X = np.array(X)
y = np.array(y)

X_test_posi = X[9500:10000]
X_test_nega = X[19500:20000]

X_test = np.vstack((X_test_posi, X_test_nega))
y_test = []
for i in range (0,500):
    y_test.append(1)

for i in range (0,500):
    y_test.append(0)
y_test = np.array(y_test)


train_numbers = 1000
apparent_error = []
true_error = []

for lamda in [0,1,10,100,1000,10000]:
    current_sum_apparent_error = []
    current_sum_true_error = []
    for i in range (0,7):

        X_train_posi = X[i*train_numbers:(i+1)*train_numbers]
        X_train_nega = X[10000+i*train_numbers:10000+(i+1)*train_numbers]
        X_train = np.vstack((X_train_posi, X_train_nega))

        y_train_posi = y[i*train_numbers:(i+1)*train_numbers]
        y_train_nega = y[10000+i*train_numbers:10000+(i+1)*train_numbers]
        y_train = np.concatenate((y_train_posi, y_train_nega))

        nmc = nearstMeanClassifer(dimension=21, learning_rate=0.01, lamda=lamda)
        nmc.fit(X_train, y_train)

        now_apparent_error, now_true_error = nmc.evaluate(X_test, y_test, X_train, y_train)
        current_sum_apparent_error.append(now_apparent_error)
        current_sum_true_error.append(now_true_error)

    apparent_error.append(np.mean(current_sum_apparent_error))
    true_error.append(np.mean(current_sum_true_error))

np.save('apparent_error_for_1000.npy', apparent_error)
np.save('true_error_for_1000.npy', true_error)

print(apparent_error)
print(true_error)



# train_numbers = 1000
# lamda = 10000
# medians_m_posi_sum = []
# medians_m_nega_sum = []
# for i in range (0,7):
#
#         X_train_posi = X[i*train_numbers:(i+1)*train_numbers]
#         X_train_nega = X[10000+i*train_numbers:10000+(i+1)*train_numbers]
#         X_train = np.vstack((X_train_posi, X_train_nega))
#
#         y_train_posi = y[i*train_numbers:(i+1)*train_numbers]
#         y_train_nega = y[10000+i*train_numbers:10000+(i+1)*train_numbers]
#         y_train = np.concatenate((y_train_posi, y_train_nega))
#
#         nmc = nearstMeanClassifer(dimension=21, learning_rate=0.01, lamda=lamda)
#         nmc.fit(X_train, y_train)
#         medians_m_posi_sum.append(nmc.m_posi)
#         medians_m_nega_sum.append(nmc.m_nega)
#
# print (medians_m_posi_sum)
# medians_m_posi = np.sum(medians_m_posi_sum,axis=0)/7
# medians_m_nega = np.sum(medians_m_nega_sum,axis=0)/7
#
# medians = np.vstack((medians_m_posi,medians_m_nega))
# np.save('medians_10000_1000.npy', medians)
# print(medians)











