import numpy as np
from sklearn.linear_model import LinearRegression

class co_training:

    def __init__(self):

        self.model_1 = LinearRegression()
        self.model_2 = LinearRegression()

    def fit(self, X, y, unlabeled):

        L1 = np.copy(X[:, 0:5])
        L1_labels = np.copy(y)

        L2 = np.copy(X[:, 6:12])
        L2_labels = np.copy(y)

        current_unlabeled = np.copy(unlabeled)

        while True:
            self.model_1.fit(L1, L1_labels)
            self.model_2.fit(L2, L2_labels)

            if len(current_unlabeled) == 0:
                break

            score_1 = self.model_1.predict(current_unlabeled[:, 0:5])

            max_index_1 = np.argmax(np.abs(score_1))
            max_score_1 = score_1[max_index_1]
            max_score_1 = int(np.sign(max_score_1))

            score_2 = self.model_2.predict(current_unlabeled[:, 6:12])

            max_index_2 = np.argmax(np.abs(score_2))
            max_score_2 = score_2[max_index_2]
            max_score_2 = int(np.sign(max_score_2))

            current__delete_set = {max_index_1, max_index_2}

            L1 = np.vstack((L1, current_unlabeled[max_index_1, 0:5]))
            L1_labels = np.concatenate((L1_labels, [max_score_1]))

            L2 = np.vstack((L2, current_unlabeled[max_index_2, 6:12]))
            L2_labels = np.concatenate((L2_labels, [max_score_2]))

            current_remain_list = list(set(range(0, len(current_unlabeled))) - current__delete_set)

            current_unlabeled = current_unlabeled[current_remain_list]

    def predict(self, X):
        score_1 = self.model_1.predict(X[:,0:5])
        score_2 = self.model_2.predict(X[:,6:12])

        score = (score_1+score_2)/2
        return score









class co_training_2D:

    def __init__(self):

        self.model_1 = LinearRegression()
        self.model_2 = LinearRegression()

    def fit(self, X, y, unlabeled):

        L1 = np.copy(X[:, 0:1])
        L1_labels = np.copy(y)

        L2 = np.copy(X[:, 1:2])
        L2_labels = np.copy(y)

        current_unlabeled = np.copy(unlabeled)

        while True:
            self.model_1.fit(L1, L1_labels)
            self.model_2.fit(L2, L2_labels)

            if len(current_unlabeled) == 0:
                break

            score_1 = self.model_1.predict(current_unlabeled[:, 0:1])
            max_index_1 = np.argmax(np.abs(score_1))

            max_score_1 = score_1[max_index_1]

            max_score_1 = int(np.sign(max_score_1))

            score_2 = self.model_2.predict(current_unlabeled[:, 1:2])

            max_index_2 = np.argmax(np.abs(score_2))

            max_score_2 = score_2[max_index_2]

            max_score_2 = int(np.sign(max_score_2))

            current__delete_set = {max_index_1, max_index_2}

            L1 = np.vstack((L1, current_unlabeled[max_index_1, 0:1]))
            L1_labels = np.concatenate((L1_labels, [max_score_1]))

            L2 = np.vstack((L2, current_unlabeled[max_index_2, 1:2]))
            L2_labels = np.concatenate((L2_labels, [max_score_2]))

            current_remain_list = list(set(range(0, len(current_unlabeled))) - current__delete_set)

            current_unlabeled = current_unlabeled[current_remain_list]

    def predict(self, X):
        score_1 = self.model_1.predict(X[:,0:1])
        score_2 = self.model_2.predict(X[:,1:2])

        score = (score_1+score_2)/2
        return score



