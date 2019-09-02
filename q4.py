import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from self_learning import self_learning
from co_training import co_training_2D

# First_set_up
# points_positive = np.random.multivariate_normal(mean = [1, 1], cov = [[2, -1.8], [-1.8, 2]], size = 500)
#
# points_negative = np.random.multivariate_normal(mean= [0, 0], cov = [[1, -0.9], [-0.9, 1]], size= 500)
#
#
# plt.scatter(points_positive[:, 0], points_positive[:, 1], label = 'Positive')
# plt.scatter(points_negative[:, 0], points_negative[:, 1], label = 'Negative')
# plt.title('Classes Distribution for Scenario 1 ')
# plt.legend()
# plt.show()
#
#
# vectors = np.vstack((points_positive, points_negative))
# labels = np. concatenate((np.ones(500, dtype=int), -np.ones(500, dtype=int)))

# vectors, labels = shuffle(vectors, labels)
points_positive_1 = np.random.multivariate_normal(mean=[0, 0], cov=[[0.1,0, ], [0, 0.1]], size=470)
points_positive_2 = np.random.multivariate_normal(mean= [1, 1 ], cov = [[0.2, 0], [0, 0.2]], size= 30)
points_negative_1 = np.random.multivariate_normal(mean= [2, 2 ], cov = [[0.1, 0], [0, 0.1]], size= 250)
points_negative_2= np.random.multivariate_normal(mean= [6, 0], cov = [[0.1, 0], [0, 0.1]], size= 250)


plt.scatter(points_positive_1[:, 0], points_positive_1[:, 1], c='g', label = 'Positive')
plt.scatter(points_positive_2[:, 0], points_positive_2[:, 1],c = 'g')
plt.scatter(points_negative_1[:, 0], points_negative_1[:, 1],c = 'b',label = 'Negative')
plt.scatter(points_negative_2 [:, 0], points_negative_2 [:, 1],c= 'b')
plt.title('Classes Distribution for Scenario 2 ')
plt.legend()
plt.show()

vectors = np.vstack((points_positive_1, points_positive_2, points_negative_1 ,points_negative_2))
labels = np. concatenate((np.ones(500, dtype=int), -np.ones(500, dtype=int)))

vectors, labels = shuffle(vectors, labels)
#
performance_origin_acc = []
#
for i in range(0, 100):
    train_positive_index = np.random.randint(0, high=500, size = 4)
    train_negative_index = np.random.randint(500,  high=1000, size = 4)
    train_index = np.concatenate((train_positive_index, train_negative_index))
    current_train_vectors = vectors[train_index]
    current_train_labels = labels[train_index]

    test_index = list(set(range(0, 1000)) - set(train_index))
    current_test_vectors = vectors[test_index]
    current_test_labels = labels[test_index]

    lr = LinearRegression()
    lr.fit(current_train_vectors, current_train_labels)
    pred = lr.predict(current_test_vectors)
    binary_pred = np.sign(pred)

    acc_score = 1-accuracy_score(current_test_labels, binary_pred)

    performance_origin_acc.append(acc_score)

print(np.mean(performance_origin_acc))

print('here')
for power in range(3, 10):

    number_of_unlabeled = np.power(2, power)
    current_performance = []
    for i in range(0, 30):
        train_positive_index = np.random.randint(0, high=500, size = 4)
        train_negative_index = np.random.randint(500,  high=1000, size = 4)
        train_index = np.concatenate((train_positive_index, train_negative_index))
        current_train_vectors = vectors[train_index]
        current_train_labels = labels[train_index]

        remain_index = list(set(range(0, 1000)) - set(train_index))
        current_remain_vectors = vectors[remain_index]
        current_remain_labels = labels[remain_index]

        current_remain_vectors, current_remain_labels = shuffle(current_remain_vectors, current_remain_labels)

        support_unlabeled_index = np.random.randint(0, high=992, size = number_of_unlabeled)

        support_vectors = current_remain_vectors[support_unlabeled_index]

        test_index = list(set(range(0, 992)) - set(support_unlabeled_index))

        current_test_vectors = current_remain_vectors[test_index]
        current_test_labels = current_remain_labels[test_index]

        lr = self_learning()

        lr.fit(current_train_vectors, current_train_labels, support_vectors)
        pred = lr.predict(current_test_vectors)
        pred = np.sign(pred)

        score = 1-accuracy_score(current_test_labels, pred)
        current_performance.append(score)
    print(np.mean(current_performance))


print('here')

for power in range(3, 10):

    number_of_unlabeled = np.power(2, power)
    current_performance = []
    for i in range(0, 30):
        train_positive_index = np.random.randint(0, high=500, size = 4)
        train_negative_index = np.random.randint(500,  high=1000, size = 4)
        train_index = np.concatenate((train_positive_index, train_negative_index))
        current_train_vectors = vectors[train_index]
        current_train_labels = labels[train_index]

        remain_index = list(set(range(0, 1000)) - set(train_index))
        current_remain_vectors = vectors[remain_index]
        current_remain_labels = labels[remain_index]

        current_remain_vectors, current_remain_labels = shuffle(current_remain_vectors, current_remain_labels)

        support_unlabeled_index = np.random.randint(0, high=992, size = number_of_unlabeled)

        support_vectors = current_remain_vectors[support_unlabeled_index]

        test_index = list(set(range(0, 992)) - set(support_unlabeled_index))

        current_test_vectors = current_remain_vectors[test_index]
        current_test_labels = current_remain_labels[test_index]

        lr = co_training_2D()

        lr.fit(current_train_vectors, current_train_labels, support_vectors)
        pred = lr.predict(current_test_vectors)
        pred = np.sign(pred)

        score = 1-accuracy_score(current_test_labels, pred)
        current_performance.append(score)
    print(np.mean(current_performance))