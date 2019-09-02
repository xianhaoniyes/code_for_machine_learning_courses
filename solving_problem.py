import numpy as np
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from aim_loss_computing import aim_loss_computing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
# from amazing_classifier import fairnessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

datafile = open('train.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
data = []
for row in datareader:
    data.append(row)

real_data = []
for row in data:
    real_data_row = []
    for item in row:
        real_data_row.append(int(float(item)))
    real_data.append(real_data_row)

real_data = np.array(real_data)


# for i in range(0,14):
#     print(np.min(real_data[:, i]), np.max(real_data[:, i]))
#

# k = 0
# for i in range(0, len(real_data)):
#     if (real_data[i, 14]==2 and real_data[i,12] == 2):
#         k = k+1
#
# print(k)

# nuermerical_fatures = real_data[:, 0:6]
#
# pca = PCA(n_components=2)
# pca.fit(nuermerical_fatures)
#
# nuermerical_fatures = pca.transform(nuermerical_fatures)
#
# plot.scatter(nuermerical_fatures[:,0],nuermerical_fatures[:,1])
# plot.show()
# labels = real_data[:,14]
# features  = real_data[:, 13]
# cate_0 = features[labels == 1]
# cate_1 = features[labels ==2]
# sns.distplot(cate_0)
# sns.distplot(cate_1)
#
# plot.show()

real_data[:, 12] = real_data[:, 12] - 1
real_data[:, 14] = real_data[:, 14] - 1
#
#
data = real_data[:, 0:13]
labels = real_data[:, 14]

# print(len(labels[labels == 0]), len(labels[labels==1]))

#
#
#
cata_one_data = data[labels == 0]
cata_two_data = data[labels == 1]



skf = StratifiedKFold (n_splits=4)

index_for_cata_one = skf.split(cata_one_data, cata_one_data[:, 12])
index_for_cata_two = skf.split(cata_two_data, cata_two_data[:, 12])

index_for_cata_one = list(index_for_cata_one)
index_for_cata_two = list(index_for_cata_two)

# print(np.shape(index_for_cata_one))
acc = []
lo = []
p_lo = []
for i in range(0, 4):

    index_train_cata_one, index_test_cata_one = index_for_cata_one[i][0],  index_for_cata_one[i][1]
    index_train_cata_two, index_test_cata_two = index_for_cata_two[i][0], index_for_cata_two[i][1]

    train_cata_one = cata_one_data[index_train_cata_one]
    test_cata_one = cata_one_data[index_test_cata_one]

    train_cata_two = cata_two_data[index_train_cata_two]
    test_cata_two = cata_two_data[index_test_cata_two]

    train_vectors = np.vstack((train_cata_one, train_cata_two))
    train_labels = np.concatenate((np.zeros(len(train_cata_one)), np.ones(len(train_cata_two))))

    train_features_13 = train_vectors[:, 12]


    weights = np.concatenate((np.ones(len(train_cata_one)), np.ones(len(train_cata_two))))

    test_vectors = np.vstack((test_cata_one, test_cata_two))
    test_labels = np.concatenate((np.zeros(len(test_cata_one)), np.ones(len(test_cata_two))))

    test_features_13 = test_vectors[:, 12]


    # print(len(train_vectors[train_labels==0]),len(train_vectors[train_labels ==1]))
    # print(len(test_vectors[test_labels == 0]), len(test_vectors[test_labels == 1]))
    # print(len(train_vectors[train_features_13 == 0]), len(train_vectors[train_features_13 == 1]))
    # print(len(test_vectors[test_features_13 == 0]), len(test_vectors[test_features_13 == 1]))





    # nb = GaussianNB()
    # nb.fit(train_vectors[:, 2:4], train_labels, sample_weight=weights)
    #
    # pre = nb.predict(test_vectors[:, 2:4])
    # pre_prob = nb.predict_proba(test_vectors[:, 2:4])
    #
    # z = 0
    # real_z = 0
    # for k in range(0,len(test_labels)):
    #     if test_labels[k] ==2 and pre[k]==2:
    #         z = z+1
    #
    # for k in range(0,len(test_labels)):
    #     if test_labels[k] ==2:
    #         real_z = real_z+1
    #
    # print(z/real_z)
    # kk = 0
    # gg = 0


    # print(np.shape(pre_prob))

    # for item in pre_prob:
    #     print(item[0]-item[1])

    # wrong_pre = []
    # for j in range(0, len(test_labels)):
    #     if(test_labels[j] != pre[i]):
    #         wrong_pre.append(np.abs(pre_prob[j, 0]-pre_prob[j, 1]))
    #
    # print(np.mean(wrong_pre))
    # #
    # right_pre = []
    # for j in range(0, len(test_labels)):
    #     if(test_labels[j] == pre[i]):
    #         right_pre.append(np.abs(pre_prob[j, 0]-pre_prob[j, 1]))
    #
    # print(np.mean(right_pre))
    #

    ss = StandardScaler()

    ss_train_vectors = train_vectors[:, 0:13]
    ss_test_vectors = test_vectors[:, 0:13]
    # ss_train_vectors = ss.fit(ss_train_vectors).transform(ss_train_vectors)
    # ss_test_vectors = ss.transform(ss_test_vectors)

    # support_train_vectors = train_vectors[:, 6:]
    # support_test_vectors = test_vectors[:, 6:]



    # pca = PCA(n_components=3)
    #
    # pca.fit(ss_train_vectors)
    #
    # ss_train_vectors = pca.transform(ss_train_vectors)
    # ss_test_vectors = pca.transform(ss_test_vectors)

    # print(np.shape(test_vectors))
    # fr = fairnessClassifier()
    # lol = fr.fit(ss_train_vectors, train_labels, train_features_13, ss_test_vectors, test_labels, test_features_13, weights)

    # svc =SVC()
    # svc.fit(ss_train_vectors, train_labels, sample_weight=weights)
    # pre =svc.predict(ss_test_vectors)
    # pre = ss_test_vectors.dot(beta[0:-1]) +beta[-1]
    # print(np.shape(pre))


    boost = GradientBoostingClassifier(n_estimators=300)
    boost.fit(ss_train_vectors, train_labels)
    # pre = boost.predict(test_vectors)

    #
    pre = boost.predict_proba(ss_test_vectors)[:, 1]
    pre_cate_one = pre[test_features_13 == 0]
    labels_cate_one = test_labels[test_features_13 == 0]
    #
    #
    cate_one_FPR, cate_one_TPR, threshold_one = roc_curve(labels_cate_one, pre_cate_one)
    #
    pre_cate_two = pre[test_features_13 == 1]
    labels_cate_two = test_labels[test_features_13 == 1]

    # cate_two_FPR, cate_two_TPR , threshold_two = roc_curve(labels_cate_two, pre_cate_two)
    #
    #
    # plot.plot(cate_one_FPR, cate_one_TPR, label = 'category one')
    # plot.plot(cate_two_FPR, cate_two_TPR, label = 'category two')
    # plot.xlabel('FPR')
    # plot.ylabel('TPR')
    # plot.legend()
    # plot.show()
    #
    #
    # plot.clf()
    # cate_one_FNR = 1- cate_one_TPR
    # cate_two_FNR = 1 - cate_two_TPR
    #
    # plot.plot(cate_one_FPR, cate_one_FNR, label = 'category one')
    # plot.plot(cate_two_FPR, cate_two_FNR, label = 'category two')
    # plot.xlabel('FPR')
    # plot.ylabel('FNR')
    # plot.legend()
    # plot.show()

    pre_cate_one [pre_cate_one >= 0.6] = 1
    pre_cate_one [pre_cate_one < 0.6] = 0

    pre_cate_two [pre_cate_two >= 0.7] = 1
    pre_cate_two [pre_cate_two < 0.7] = 0

    new_pre = np.concatenate((pre_cate_one,pre_cate_two))
    new_test_labels = np.concatenate((labels_cate_one,labels_cate_two))
    new_test_features_13 = np.concatenate((np.zeros(len(pre_cate_one)),np.ones(len(pre_cate_two))))



    # lol, p = aim_loss_computing(pre,test_labels,test_features_13)
    #
    lol , p = aim_loss_computing(new_pre, new_test_labels, new_test_features_13)
    # lol = accuracy_score(test_labels,pre)
    # print(lol)
#
    lo.append(lol)
    p_lo.append(p)
#
print(np.mean(lo), np.std(lo))
print(np.mean(p_lo, axis =0))
print(np.std(p_lo, axis= 0))

























