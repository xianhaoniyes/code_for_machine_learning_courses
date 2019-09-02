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


def extractinstances(image):

    features = []
    # resize the image to speed up clustering
    image = image.resize((128, 96))
    image = np.array(image)
    flat_image = np.reshape(image, [-1, 3])
    bandwidth = 23

    # perform Mean-shift clustering
    ms = MeanShift(bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    number_of_segment = len(ms.cluster_centers_)
    labels = ms.labels_

    # Find all the segment after the clustering and extract the corresponding features

    for i in range(0, number_of_segment):
        current_segment = flat_image[labels == i]
        mean_red = np.mean(current_segment[:, 0])
        mean_green = np.mean(current_segment[:, 1])
        mean_blue = np.mean(current_segment[:, 2])
        segment_feature = [mean_red, mean_green, mean_blue]
        features.append(segment_feature)

    features = np.array(features)

    return features















