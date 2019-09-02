
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import numpy as np

class citation_KNN:

    def __init__(self, k, c):

        self.k = k
        self.c = c

    # simple save all the train_bags
    def fit(self, train_bag_instance_features, train_bag_labels):
        self.train_bag_instance_features = train_bag_instance_features
        self.train_bag_labels = train_bag_labels

    # predict the label of bags
    def predict(self, test_bag_instance_features):

        pred = []

        for item in test_bag_instance_features:

            C_citations = []

            total_bag_instance_features = []
            total_bag_instance_features.append(item)
            for  i in range(0,len(self.train_bag_instance_features)):
                total_bag_instance_features.append(total_bag_instance_features[i])

            D = self.minumum_hausdorff_silarity_matrix_form(total_bag_instance_features)
            print(D)

            # here we take the R-nearest bags
            R_neighbors = D[0, 1:self.k+1]

            for i in range(1, len(D)):
                if 0 in D[i, 0:self.c]:
                    C_citations.append(i)

            # consider the bags which will also rank the target(test) bag to a high place when consider distance.
            C_citations = np.array(C_citations)

            positive = 0
            negative = 0

            # how many apple in the (R+C) bags are apples or bananas, take the label of the dominate one as the predict
            # label
            for item_x in R_neighbors:
                if self.train_bag_labels[item_x-1] == 1:
                    positive = positive+1
                else:
                    negative = negative+1

            for item_x in C_citations:
                if self.train_bag_labels[item_x - 1] == 1:
                    positive = positive + 1
                else:
                    negative = negative + 1

            if positive >= negative:
                pred.append(1)
            else:
                pred.append(0)

        pred = np.array(pred)
        return pred

    # for each bags, return the sort of other bags based on their distance to the target bag (nearest neighbors)
    def minumum_hausdorff_silarity_matrix_form(self, total_bag_instance_features):

        simi = np.zeros((len(total_bag_instance_features),len(total_bag_instance_features)), dtype=int)
        D = np.zeros((len(total_bag_instance_features),len(total_bag_instance_features)))

        for i in range(0,len(total_bag_instance_features)):
            for j in range(i+1,len(total_bag_instance_features)):
                D[i,j] = self.minimum_hausdorff_dinstance(total_bag_instance_features[i],
                                                          total_bag_instance_features[j])

        D = D + np.transpose(np.triu(D, k=1))


        for i in range(0, len(simi)):
            simi[i, :] = np.argsort(D[i, :])

        return simi

    # calculate the distance between two bags
    def minimum_hausdorff_dinstance(self, bag_1, bag_2):
        current_minimum = 1000.0
        for item_1 in bag_1:
            for item_2 in bag_2:
                distance = np.linalg.norm(item_1-item_2)

                if current_minimum>distance:
                    current_minimum = distance

        return current_minimum


bag_labels = np.load('bag_labels.npy')
bag_instance_features = np.load('bag_instance_features.npy')

skf = StratifiedKFold(n_splits=4)
acc_total = []
for train_index, test_index in skf.split(bag_instance_features, bag_labels):
    bag_train_labels, bag_train_features = bag_labels[train_index], bag_instance_features[train_index]
    bag_test_labels, bag_test_features = bag_labels[test_index], bag_instance_features[test_index]


    solver = citation_KNN(k=3,c = 7)

    solver.fit(bag_train_features, bag_train_labels)

    pred = solver.predict(bag_test_features)

    acc = accuracy_score(bag_test_labels, pred)

    acc_total.append(acc)


print(np.mean(acc_total))









