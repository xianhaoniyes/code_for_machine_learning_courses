import numpy as np


def aim_loss_computing(pred, true, features_13):


    p_2_11 = len(list(filter(lambda i: pred[i] == 1 and true[i] == 0 and features_13[i] == 0, range(0, len(pred)))))/\
             len(list(filter(lambda i: true[i] == 0 and features_13[i] == 0, range(0, len(pred)))))

    p_2_12 = len(list(filter(lambda i: pred[i] == 1 and true[i] == 0 and features_13[i] == 1, range(0, len(pred)))))/ \
             len(list(filter(lambda i: true[i] == 0 and features_13[i] == 1, range(0, len(pred)))))

    p_1_21 = len(list(filter(lambda i: pred[i] == 0 and true[i] == 1 and features_13[i] == 0, range(0, len(pred)))))/\
             len(list(filter(lambda i: true[i] == 1 and features_13[i] == 0, range(0, len(pred)))))

    p_1_22 = len(list(filter(lambda i: pred[i] == 0 and true[i] == 1 and features_13[i] == 1, range(0, len(pred)))))/\
             len(list(filter(lambda i: true[i] == 1 and features_13[i] == 1, range(0, len(pred)))))

    # print(p_2_11, p_2_12, p_1_21, p_1_22)

    # print(3*np.max((p_2_11, p_2_12)) + np.max((p_1_21,p_1_22)))

    return  3*np.max((p_2_11, p_2_12)) + np.max((p_1_21, p_1_22)), [p_2_11, p_2_12, p_1_21, p_1_22 ]





