import numpy as np
import sklearn
import matplotlib.pyplot as plot
# quesion b,c
class weak_learn:

    def __init__(self):
        self. feature = 0
        self. threshold = 0
        self.judgement = 0

# exhaustive search on the training set for the optimum feature and threshold
# for each feature feature dimension the threshold start with the minimum feature value and end with the largest
    def fit(self, X, y, gran, weight_vector):
        current_feature = 0
        current_threshold =0
        current_judgement = 0
        current_app_error = 1
        dimension = len(X[0])
        weight_vector = weight_vector/np.sum(weight_vector)
        for i in range (0,dimension):
            threshold_lower_bound = np.min(X[:,i])
            threshold_upper_bound = np.max(X[:,i])

            theta = threshold_lower_bound
            step = (threshold_upper_bound-threshold_lower_bound)/gran
            while theta <= threshold_upper_bound:

                current_result, judgement = self.fit_predict(theta, X[:, i], y, weight_vector)

                if current_result < current_app_error:
                    current_feature = i
                    current_threshold = theta
                    current_app_error = current_result
                    current_judgement = judgement

                theta = theta + step

        self.feature = current_feature
        self.threshold = current_threshold
        self.judgement = current_judgement
        return current_app_error

# calculate the current weighted error for current feature and threshold
    def fit_predict(self,current_threshhold, x, y, weight_vector):
        resullt_1 = np.empty([len(y), ], dtype=int)
        resullt_2 = np.empty([len(y), ], dtype=int)
        miss_1 = 0
        miss_2 = 0

        for i in range(0, len(y)):
            if x[i] <= current_threshhold:
                resullt_1[i] = 0
            else:
                resullt_1[i] = 1
            if resullt_1[i] != y[i]:
                miss_1 = miss_1 + weight_vector[i]

            if x[i] >= current_threshhold:
                resullt_2[i] = 0
            else:
                resullt_2[i] = 1
            if resullt_2[i] != y[i]:
                miss_2 = miss_2 + weight_vector[i]

        if miss_1 <= miss_2:
            return miss_1, 0
        else:
            return miss_2, 1

    def predict(self,X):
        prediction = np.empty((len(X)),dtype=int)
        if self.judgement == 0:
            for i in range(0,len(X)):

                if X[i][self.feature]<=self.threshold:
                    prediction[i] = 0
                else:
                    prediction[i] = 1

        else:
            for i in range(0, len(X)):
                if X[i][self.feature] >= self.threshold:

                    prediction[i] = 0
                else:
                    prediction[i] = 1

        return prediction

    def evaluate(self,X,y):
        prediction = self.predict(X)
        miss = 0
        for i in range(0, len(y)):
            if prediction[i] != y[i]:
                miss = miss+1

        return miss/len(y)


    def get_parameter(self):
        print(self.feature, self.threshold)



