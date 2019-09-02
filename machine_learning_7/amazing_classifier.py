from __future__ import division
import cvxpy as cp
import numpy as np
import dccp
from aim_loss_computing import aim_loss_computing
class fairnessClassifier():

    def _init_(self,):
        self.beta = 0

    def fit(self, train_vectors, train_labels, train_features_13, test_vectors, test_labels, test_features_13, weights):
        c0 = 100
        c1 = 10
        c2 = 10

        train_labels = np.expand_dims(train_labels, axis=1)
        n = len(train_vectors[0])

        beta = cp.Variable((n+1, 1))

        # lambd = cp.Parameter(nonneg=True)
        lambd = 1

        Y = train_labels

        X = train_vectors
        X = cp.hstack([X, np.ones((len(X),1))])

        N = len(train_labels)

        N0 = len(train_labels[train_features_13 == 0])
        N1 = len(train_labels[train_features_13 == 1])


        x0 = X[train_features_13 == 0]
        y0 = Y[train_features_13 == 0]

        x1 = X[train_features_13 == 1]
        y1 = Y[train_features_13 == 1]

        # log_likelihood = cp.sum(
        #     (-cp.reshape(cp.multiply(Y, X @ beta), (len(Y),)) +
        #     cp.log_sum_exp(cp.hstack([np.zeros((len(Y), 1)), X @ beta]), axis=1))) + \
        #     lambd * cp.norm(beta[0:-1], 2)


        # log_likelihood = cp.sum(
        #     (cp.multiply(-cp.reshape(cp.multiply(Y, X @ beta), (len(Y),)) +
        #     cp.log_sum_exp(cp.hstack([np.zeros((len(Y), 1)), X @ beta]), axis=1), weights))) + \
        #     lambd * cp.norm(beta[0:-1], 2)

        log_likelihood = cp.sum(cp.multiply(cp.reshape(cp.logistic(cp.multiply(-Y, X@beta)), (len(Y),)), weights))\
                         + lambd * cp.norm(beta[0:-1], 2)



        # print(np.shape(np.zeros((len(y0), 1))))

        problem = cp.Problem(cp.Minimize(log_likelihood),

        [

            # (N1 / float(N)) * cp.sum(cp.minimum(
            # np.zeros((len(y0), 1)), cp.multiply(y0, (x0 @ beta)))) \
            # >= (N0 / float(N)) * cp.sum(cp.minimum(
            # np.zeros((len(y1), 1)), cp.multiply(y1, (x1 @ beta))))-c0,
            #
            # (N1 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y0), 1)), cp.multiply(y0, (x0 @ beta)))) \
            # <= (N0 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y1), 1)), cp.multiply(y1, (x1 @ beta)))) + c0,
            #
            #
            #
            # (N1 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y0), 1)), cp.multiply((1-y0)/2*y0, (x0 @ beta)))) \
            # >= (N0 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y1), 1)), cp.multiply((1-y1)/2*y1, (x1 @ beta)))) - c1,
            #
            # (N1 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y0), 1)), cp.multiply((1-y0)/2*y0, (x0 @ beta)))) \
            # <= (N0 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y1), 1)), cp.multiply((1-y1)/2*y1, (x1 @ beta)))) + c1,
            #
            #
            #
            # (N1 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y0), 1)), cp.multiply((1 + y0) / 2 * y0, (x0 @ beta)))) \
            # >= (N0 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y1), 1)), cp.multiply((1 + y1) / 2 * y1, (x1 @ beta)))) - c2,
            #
            # (N1 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y0), 1)), cp.multiply((1 + y0) / 2 * y0, (x0 @ beta)))) \
            # <= (N0 / float(N)) * cp.sum(cp.minimum(
            #     np.zeros((len(y1), 1)), cp.multiply((1 + y1) / 2 * y1, (x1 @ beta)))) + c2


            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply(y0, (x0 @ beta)))) \
            >= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply(y1, (x1 @ beta))))-c0,

            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply(y0, (x0 @ beta)))) \
            <= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply(y1, (x1 @ beta)))) + c0,


            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1-y0)/2.0, cp.multiply(y0, x0 @ beta)))) \
            >= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1-y1)/2.0, cp.multiply(y1, x1 @ beta)))) - c1,

            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1-y0)/2.0, cp.multiply(y0, x0 @ beta)))) \
            <= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1-y1)/2.0,cp.multiply(y1, x1 @ beta)))) + c1,



            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1 + y0) / 2.0, cp.multiply( y0, x0 @ beta)))) \
            >= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1 + y1) / 2.0 , cp.multiply(y1, x1 @ beta)))) - c2,

            (N1 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1 + y0) / 2.0,cp.multiply( y0, x0 @ beta)))) \
            <= (N0 / float(N)) * cp.sum(cp.minimum(
            0, cp.multiply((1 + y1) / 2.0 ,cp.multiply( y1, x1 @ beta)))) + c2

         ])

        problem.solve(method='dccp')

        print(beta.value)

        self.beta = beta.value
        loss = self.predict(test_vectors, test_labels, test_features_13)
        print(loss)
        return loss


    def predict(self,test_vectors,test_labels,test_feautures_13):

        pred = test_vectors.dot(self.beta[0:-1]) + self.beta[-1]

        pred = 1/(1+np.exp(-pred))

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = -1

        loss = aim_loss_computing(pred, test_labels, test_feautures_13)

        return loss








