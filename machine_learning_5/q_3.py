from q_iteration import q_iteration
from q_learning import q_learning
import numpy as np
import matplotlib.pyplot as plt

epsilon_value = [0.4, 0.6, 0.8]
learning_rate_value = [0.4, 0.6, 0.8]
maximum_iterations = [0,3000,6000, 100000, 200000, 300000, 400000, 500000, 600000]
gamma = 0.9


q_iter = q_iteration(gamma=gamma)
q_iter.fit()
q_iter_v = q_iter.v_value()
total_difference = []

for epsilon in epsilon_value:
    for learning_rate in learning_rate_value:
            differences = []
            for maximum_iteration in maximum_iterations:
                q_learn = q_learning(gamma=gamma, epsilon=epsilon, learning_rate = learning_rate)
                q_learn.fit(maximum_iterations=maximum_iteration, maximum_per_eposide=5000)
                q_learn_v = q_learn.v_value()

                diff = np.linalg.norm(q_iter_v-q_learn_v)
                print(diff)

                differences.append(diff)
            total_difference.append(differences)



np.save('total_differences.npy', total_difference)
total_difference = np.load('total_differences.npy')
k = 0
for i in range(0, len(epsilon_value)):
    for j in range(0, len(learning_rate_value)):
        plt.plot(maximum_iterations, total_difference[k],
                 label='epsilon = ' + str(epsilon_value[i]) + ' learning_rate = ' + str(learning_rate_value[j]))
        k = k+1


plt.legend()
plt.ylabel('norm2 distance')
plt.xlabel('iterations')
plt.show()






