import numpy as np
import sklearn
from weak_learn import weak_learn
import math


class adaboost:

    def __init__(self):
        self.models = []
        self.classifier_weights = []

    #training the model
    def fit(self, X, y, gran,T):

        for i in range(0,T):
            self.models.append(weak_learn())

        sample_weights = np.ones((len(y),))
        #training the model up to T iterations
        for i in range (0,T):

            model = self.models[i]
            error = model.fit(X, y, gran=gran, weight_vector=sample_weights)
            if error == 0:
                beta = 0.0001
            else:
                beta = error/(1-error)
            sample_weights = self.special_func(X, y, beta, sample_weights, model)
            self.classifier_weights.append(math.log(1/beta))

        return sample_weights

    # this function is for calculationg new_sample_weights
    def special_func(self,X,y,beta,sample_weights,model):

        imm = 1-abs(model.predict(X)-y)
        sample_weights = sample_weights*np.power(beta,imm)
        return sample_weights

    # function for predicting after training
    def predict(self,X):

        results_x = np.zeros(len(X))
        # calculating results for each sub classifiers and add them together
        for i in range(0,len(self.classifier_weights)):

            results_x = results_x + self.classifier_weights[i]*(self.models[i].predict(X)-0.5)
        for i in range(0, len(results_x)):

            if results_x[i] >= 0:
                results_x[i] = 1
            else:
                results_x[i] = 0

        return results_x


    def evaluate(self,X,y):
        if self.models[0] is self.models[1]:
            print("the weak")
        prediction = self.predict(X)
        miss = 0
        for i in range(0, len(y)):
            if prediction[i] != y[i]:
                miss = miss+1

        return miss/len(y)








