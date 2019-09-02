import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle
from PIL import Image
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from extractinstances import extractinstances
import time



def gendatmilsival():

    instance_labels = []
    bag_instance_features = []
    bag_labels = []

    for filename in glob.glob('apple/*.jpg'):

        image = Image.open(filename)
        image_segments_features = extractinstances(image)

        bag_labels.append(1)
        instance_labels.append(np.ones(len(image_segments_features), dtype=int))
        bag_instance_features.append(image_segments_features)

    for filename in glob.glob('banana/*.jpg'):
        image = Image.open(filename)
        image_segments_features = extractinstances(image)

        bag_labels.append(0)
        instance_labels.append(np.zeros(len(image_segments_features), dtype=int))
        bag_instance_features.append(image_segments_features)

    instance_labels = np.concatenate(instance_labels)
    instance_features = np.vstack(bag_instance_features)


    # store the instance features and labels
    np.save('instance_labels.npy', instance_labels)
    np.save('instance_features.npy', instance_features)

    # store the instance feautres and labels by the category of bags
    np.save('bag_instance_features.npy', bag_instance_features)
    np.save('bag_labels.npy', bag_labels)