import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from PIL import Image
import glob
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from combineinstlabels import combineinstlabels
import time


def bagembed():

    instance_features = np.load('instance_features.npy')
    bag_instance_features = np.load('bag_instance_features.npy')
    lam = 7
    bag_features = []

    for i in range(0, len(bag_instance_features)):

        current_bag_instance_features = bag_instance_features[i]
        feature = []

        for n in range(0, len(instance_features)):

            s = np.max(np.exp(-np.linalg.norm((current_bag_instance_features-instance_features[n]), axis=1)/(lam*lam)))
            feature.append(s)

        bag_features.append(feature)

    bag_features = np.array(bag_features)
    print(np.shape(bag_features))
    np.save('bag_features.npy', bag_features)


bagembed()

labels = np.load('bag_labels.npy')
features = np.load('bag_features.npy')

#############################################################################

skf =StratifiedKFold(n_splits=4)

total_acc = []

for train_index, test_index in skf.split(features, labels):
    train_features, train_labels = features[train_index], labels[train_index]
    test_features, test_labels = features[test_index], labels[test_index]

    svc = LinearSVC(penalty= 'l1', dual = False, max_iter=100000)

    svc.fit(train_features,train_labels)

    pres = svc.predict(test_features)
    acc = accuracy_score(test_labels,pres)

    total_acc.append(acc)

print(np.mean(total_acc))












