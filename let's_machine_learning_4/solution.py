import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from self_learning import self_learning
from co_training import co_training
from sklearn.metrics import mean_squared_error
# for question 2
vectors = np.load('vectors.npy')
labels = np.load('labels.npy')
print(np.shape(vectors))
vectors, labels = shuffle(vectors, labels)

#calculate the supervised error rates
performance_origin_acc = []
performance_origin_square = []
for i in range(0, 100):
    train_positive_index = np.random.randint(0, high=5000, size = 8)
    train_negative_index = np.random.randint(5000,  high=10000, size = 8)
    train_index = np.concatenate((train_positive_index, train_negative_index))
    current_train_vectors = vectors[train_index]
    current_train_labels = labels[train_index]

    test_index = list(set(range(0, 10000)) - set(train_index))
    current_test_vectors = vectors[test_index]
    current_test_labels = labels[test_index]

    lr = LinearRegression()
    lr.fit(current_train_vectors, current_train_labels)
    pred = lr.predict(current_test_vectors)
    binary_pred = np.sign(pred)

    acc_score = 1-accuracy_score(current_test_labels, binary_pred)
    sq_loss_score = mean_squared_error(current_test_labels, pred)
    performance_origin_acc.append(acc_score)
    performance_origin_square.append(sq_loss_score)


performance_se_acc = []
performance_se_square = []

performance_co_acc = []
performance_co_square = []
for power in range(3, 10):

    number_of_unlabeled = np.power(2, power)
    print(number_of_unlabeled)
    current_performance_se_acc = []
    current_performance_co_acc = []
    current_performance_se_square = []
    current_performance_co_square = []
    for i in range(0, 100):
        train_positive_index = np.random.randint(0, high=5000, size = 8)
        train_negative_index = np.random.randint(5000,  high=10000, size = 8)
        train_index = np.concatenate((train_positive_index, train_negative_index))
        current_train_vectors = vectors[train_index]
        current_train_labels = labels[train_index]

        remain_index = list(set(range(0, 10000)) - set(train_index))
        current_remain_vectors = vectors[remain_index]
        current_remain_labels = labels[remain_index]

        current_remain_vectors, current_remain_labels = shuffle(current_remain_vectors, current_remain_labels)

        support_unlabeled_index = np.random.randint(0, high=9984, size = number_of_unlabeled)

        support_vectors = current_remain_vectors[support_unlabeled_index]

        test_index = list(set(range(0, 9984)) - set(support_unlabeled_index))

        current_test_vectors = current_remain_vectors[test_index]
        current_test_labels = current_remain_labels[test_index]

        l_se_learning = self_learning()
        l_se_learning.fit(current_train_vectors, current_train_labels, support_vectors)
        pred = l_se_learning.predict(current_test_vectors)
        binary_pred = np.sign(pred)
        acc_score_se = 1-accuracy_score(current_test_labels, binary_pred)
        sq_loss_score_se = mean_squared_error(current_test_labels,pred)
        current_performance_se_acc.append(acc_score_se)
        current_performance_se_square.append(sq_loss_score_se)


        l_co_training = co_training()
        l_co_training.fit(current_train_vectors, current_train_labels, support_vectors)
        pred = l_co_training.predict(current_test_vectors)
        binary_pred = np.sign(pred)
        acc_score_co = 1-accuracy_score(current_test_labels, binary_pred)
        sq_loss_score_co = mean_squared_error(current_test_labels, pred)
        current_performance_co_acc.append(acc_score_co)
        current_performance_co_square.append(sq_loss_score_co)

    performance_se_acc.append(current_performance_se_acc)
    performance_co_acc.append(current_performance_co_acc)

    performance_se_square.append(current_performance_se_square)
    performance_co_square.append(current_performance_co_square)


performance_se_acc = np.vstack((performance_origin_acc, performance_se_acc))
performance_se_square = np.vstack((performance_origin_square,performance_se_square))

performance_co_acc = np.vstack((performance_origin_acc, performance_co_acc))
performance_co_square = np.vstack((performance_origin_square, performance_co_square))


se_acc = np.mean(performance_se_acc, axis=1)

se_square = np.mean(performance_se_square, axis=1)

co_acc = np.mean(performance_co_acc, axis=1)
co_square = np.mean(performance_co_square, axis=1)

# np.save('se_acc.npy', se_acc)
# np.save('se_square.npy', se_square)
#
# np.save('co_acc.npy', co_acc)
# np.save('co_square.npy', co_square)

