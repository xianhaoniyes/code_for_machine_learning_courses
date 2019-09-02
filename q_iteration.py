import numpy as np
import matplotlib.pyplot as plt
#
#
class q_iteration:

    def __init__(self, gamma):

        self.gamma = gamma

        self.rewards = np.zeros(63)

        self.rewards[1*7+3] = 1

        self.actions_weights = np.zeros((63, 4))


    def fit(self):

        number_of_sweeps = 0

        while True:
            changine =False
            for i in range(0, 9):
                for j in range(0, 7):
                    if i in [0, 8] or j in [0, 6] or (i == 2 and j in [2, 3, 5]) or (i == 4 and j in [2, 3, 4]) or (i == 6 and
                    j in [2, 3, 5]) or (i == 1 and j == 3):
                        continue
                    else:
                        former_0 = self.actions_weights[7*i+j, 0]
                        former_1 = self.actions_weights[7*i+j, 1]
                        former_2 = self.actions_weights[7*i+j, 2]
                        former_3 = self.actions_weights[7*i+j, 3]

                        self.actions_weights[7*i+j, 0] = self.rewards[7*(i-1)+j] + self.gamma*np.max(self.actions_weights[7*(i-1)+j])
                        self.actions_weights[7*i+j, 1] = self.rewards[7*i+j+1] + self.gamma*np.max(self.actions_weights[7*i+j+1])
                        self.actions_weights[7*i+j, 2] = self.rewards[7*(i+1)+j] + self.gamma*np.max(self.actions_weights[7*(i+1)+j])
                        self.actions_weights[7*i+j, 3] = self.rewards[7*i+j-1] + self.gamma*np.max(self.actions_weights[7*i+j-1])

                        if changine == False:
                            if former_0 != self.actions_weights[7*i+j, 0] or former_1 != self.actions_weights[7*i+j, 1]\
                            or former_2 != self.actions_weights[7*i+j, 2] or former_3 != self.actions_weights[7*i+j, 3]:
                                changine = True

            number_of_sweeps = number_of_sweeps+1
            if changine == False:
                break

        print(number_of_sweeps)

    def optimal_path(self, start):

        path = []
        path.append([start[0], start[1]])
        current = start

        while current[0]*7 + current[1] != 10:

            postion = np.argmax(self.actions_weights[7*(current[0])+current[1]])

            if postion == 0:
                current[0] = current[0]-1
            if postion == 1:
                current[1] = current[1]+1
            if postion == 2:
                current[0] = current[0]+1
            if postion == 3:
                current[1] = current[1]-1

            i = current[0]
            j = current[1]

            path.append([i, j])

        return path

    def v_value(self):
        v_value = np.zeros((9, 7))
        for i in range(0,9):
            for j in range(0,7):
                v_value[i, j] = np.max(self.actions_weights[7*i+j])
        return v_value




# q = q_iteration(gamma=0.9)
# q.fit()
# print(q.optimal_path([7, 5]))
# value = q.v_value()
# plt.matshow(value)
# plt.show()







