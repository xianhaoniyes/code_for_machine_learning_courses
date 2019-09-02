import numpy as np

class q_learning:

    def __init__(self, gamma, epsilon, learning_rate):

        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        self.rewards = np.zeros(63)

        self.rewards[1 * 7 + 3] = 1

        self.actions_weights = np.zeros((63, 4))

    def fit(self, maximum_iterations, maximum_per_eposide):

        for e in range(0, maximum_iterations):
            k = 0
            initial_state = [7, 5]
            i = initial_state[0]
            j = initial_state[1]
            while 7*i+j != 10 and k < maximum_per_eposide:

                action = np.argmax(self.actions_weights[7*i+j])

                posi = np.random.random()

                if posi < self.epsilon:
                    action = np.random.random_integers(0, 3)

                if action == 0:
                    weights = self.actions_weights[7 * i + j, 0]

                    self.actions_weights[7 *i + j, 0] = \
                    weights +self.learning_rate*(self.rewards[7 * (i-1)+j] +
                    self.gamma*np.max(self.actions_weights[7 * (i-1)+j]) - weights)
                    i = i-1

                elif action == 1:
                    weights = self.actions_weights[7 * i + j, 1]

                    self.actions_weights[7 * i + j, 1] = \
                        weights + self.learning_rate * (self.rewards[7 * i + j+1] +
                        self.gamma * np.max(self.actions_weights[7 * i + j+1]) - weights)
                    j = j+1

                elif action == 2:
                    weights = self.actions_weights[7 * i + j, 2]

                    self.actions_weights[7 * i + j, 2] = \
                        weights + self.learning_rate * (self.rewards[7 * (i+ 1) + j] +
                        self.gamma * np.max(self.actions_weights[7 * (i+ 1) + j]) - weights)
                    i = i+1

                elif action == 3:
                    weights = self.actions_weights[7 * i+ j, 3]

                    self.actions_weights[7 * i + j, 3] = \
                        weights + self.learning_rate * (self.rewards[7 * i+ j - 1] +
                        self.gamma * np.max(self.actions_weights[7 * i + j - 1]) - weights)
                    j = j-1

                if i in [0, 8] or j in [0, 6] or (i == 2 and j in [2, 3, 5]) or (i == 4 and j in [2, 3, 4]) or (
                        i == 6 and
                        j in [2, 3, 5]) or (i == 1 and j == 3):
                    break
                k = k+1

    def optimal_path(self, start):

        path = []
        path.append([start[0], start[1]])
        current = start

        while current[0]*7 + current[1] != 10:
            print(current)

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
        for i in range(0, 9):
            for j in range(0, 7):
                v_value[i, j] = np.max(self.actions_weights[7*i+j])
        return v_value



# q= q_learning(gamma=0.9,epsilon=0.3, learning_rate=0.75)
# q.fit(maximum_iterations=200000, maximum_per_eposide=1000)
# print(q.optimal_path([7, 5]))