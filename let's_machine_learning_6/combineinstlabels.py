import numpy as np
def combineinstlabels(labels):
    return int(np.round(np.sum(labels)/len(labels)))