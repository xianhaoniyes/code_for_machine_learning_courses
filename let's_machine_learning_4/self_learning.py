import numpy as np
from sklearn.linear_model import LinearRegression


class self_learning:

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y, unlabeled):

        current_unlabeled = np.copy(unlabeled)
        vectors = np.copy(X)
        labels = np.copy(y)
        while True:

            self.model.fit(vectors, labels)

            if len(current_unlabeled) == 0:
                break

            score = self.model.predict(current_unlabeled)
            max_index = np.argmax(np.abs(score))
            max_score = int(np.sign(score[max_index]))

            vectors = np.vstack((vectors, current_unlabeled[max_index]))
            labels = np.concatenate((labels, [max_score]))
            current_unlabeled = np.delete(current_unlabeled, max_index, axis=0)



    def predict(self, X):
        y = self.model.predict(X)
        return y







