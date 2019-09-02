import numpy as np
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D

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

data = real_data[:, 0:14]
labels = real_data[:, 14]

ss = StandardScaler()
ss.fit(data)
data = ss.transform(data)

# cata_one_features  = data[labels== 1]
# cata_one_features = cata_one_features[:,3]
# sns.distplot(cata_one_features,label = '1')
#
# cata_two_features  = data[labels== 2]
# cata_two_features = cata_two_features[:,3]
# sns.distplot(cata_two_features,label='2')
# #
# plot.legend()
# plot.show()
#
# print(np.mean(cata_one_features))
# print(np.mean(cata_two_features))
# fig = plot.figure()
# ax = Axes3D(fig)

pca = PCA(n_components=2)
pca.fit(data)
cata_one_features = data[labels == 1]
cata_two_features = data[labels == 2]

cata_one_features = pca.transform(cata_one_features)
cata_two_features = pca.transform(cata_two_features)

plot.scatter(cata_one_features[:,0],  cata_one_features[:,1], s =1,label = '1')
plot.scatter(cata_two_features[:,0],  cata_two_features[:,1], s= 1, label = '2')
plot.legend()
plot.show()
