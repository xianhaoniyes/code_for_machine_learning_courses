import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
from combineinstlabels import combineinstlabels
from gendatmilsival import gendatmilsival



# gendatmilsival()
################################### not trustworthy implementation ###########################################

# instance_labels = np.load('instance_labels.npy')
# instance_features = np.load('instance_features.npy')
#
# bag_labels = np.load('bag_labels.npy')
# bag_instance_features = np.load('bag_instance_features.npy')
#
# lda = LinearDiscriminantAnalysis()
# lda.fit(instance_features, instance_labels)
#
# predict_label = []
#
# for i in range(0, len(bag_labels)):
#     pred = lda.predict(bag_instance_features[i])
#     predict_label.append(combineinstlabels(pred))
#
# apple_mis = 0
# banana_mis = 0
# for i in range(0, len(bag_labels)):
#     if bag_labels[i] == 1 and predict_label[i] ==0:
#         apple_mis = apple_mis+1
#     elif bag_labels[i] == 0 and predict_label[i] ==1:
#         banana_mis = banana_mis+1
#
# print(apple_mis, banana_mis, (apple_mis+banana_mis)/len(bag_labels), len(bag_labels))
#
# print(accuracy_score(bag_labels, predict_label))


#####################trust_worthy_implementation #############################################################

bag_labels = np.load('bag_labels.npy')
bag_instance_features = np.load('bag_instance_features.npy')

skf = StratifiedKFold(n_splits=4)
acc_total = []
for train_index, test_index in skf.split(bag_instance_features, bag_labels):
    bag_train_labels, bag_train_features = bag_labels[train_index], bag_instance_features[train_index]
    bag_test_labels, bag_test_features = bag_labels[test_index], bag_instance_features[test_index]

    train_labels = []

    for i in range(0, len(bag_train_features)):
        if bag_train_labels[i] == 1:
            train_labels.append(np.ones(len(bag_train_features[i]), dtype=int))
        else:
            train_labels.append(np.zeros(len(bag_train_features[i]), dtype=int))

    train_labels = np.concatenate(train_labels)
    train_features = np.vstack(bag_train_features)
    lda = LinearDiscriminantAnalysis()
    lda.fit(train_features, train_labels)

    predict_label = []

    for i in range(0, len(bag_test_labels)):
        pred = lda.predict(bag_test_features[i])
        predict_label.append(combineinstlabels(pred))

    acc = accuracy_score(bag_test_labels, predict_label)
    acc_total.append(acc)


print(np.mean(acc_total))