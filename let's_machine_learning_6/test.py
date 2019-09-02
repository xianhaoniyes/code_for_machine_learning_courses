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






# apple_list = []
# banana_list = []
# bandwidths = []
# k =0
# for filename in glob.glob('banana/*.jpg'):
#     start = time.time()
#     im = Image.open(filename)
#     print(np.shape(im))
#     # reshape the figure size, speed up the clustering speed
#     im = im.resize((128, 96))
#     image = np.array(im)
#     flat_image = np.reshape(image, [-1, 3])
#     # bandwidth = estimate_bandwidth(flat_image, quantile=.1, n_samples=500)
#     # bandwidths.append(bandwidth)
#     bandwidth =23
#     ms = MeanShift(bandwidth, bin_seeding=True)
#     ms.fit(flat_image)
#     labels = ms.labels_
    # print(np.bincount(labels))
    # print(np.max(labels))
    # ceters = ms.cluster_centers_
    # print(ceters)



    # Plot image vs segmented image
    # plt.clf()
    # plt.figure(2)
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.reshape(labels, [image.shape[0], image.shape[1]]))
    # plt.axis('off')
    # plt.show()
    # # plt.savefig(str(k))
    # # k = k+1
    # break
#
#     # end = time.time()
#     # print(end-start)

# print(np.mean(bandwidths))

a = np.load('instance_labels.npy')
# b = np.load('bag_labels.npy')
print(len(a))

# labels = np.load('instance_labels.npy')
# features = np.load('instance_features.npy')
#
#
#
# postive = features[labels ==1]
# negative = features[labels == 0]
#
# plt.scatter(postive[:, 0], postive[:, 1], label = 'apple')
# plt.scatter(negative[:, 0], negative[:, 1], label = 'banana')
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Distribution of Apple and Banana')
# plt.show()
