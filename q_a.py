import numpy as np
import math
adversary_matrix = [[0, 0, 1, 0],
                   [0.1, 0, 0, 0.9],
                   [0.2, 0.1, 0, 0]]

adversary_matrix = np.array(adversary_matrix)

#p_t value for each expert
strategy_matrix = [[1/3, 0, 0, 0],
                   [1/3, 0, 0, 0],
                   [1/3, 0, 0, 0]]

strategy_matrix = np.array(strategy_matrix)

T = 4

# question_a
#strategy_A
# for t in range (1,T):
#     L_t_1 = np.sum(adversary_matrix[:, 0:t], axis=1) # calculate cumlative loss for time t
#     L_t_1 = L_t_1.tolist()
#     strategy_matrix[L_t_1.index(min(L_t_1))][t] = 1  # find the expert with minimum loss


#stragegy_B
for t in range(1, T):
    L_t_1 = np.sum(adversary_matrix[:, 0:t], axis=1)
    e_L_matrix = np.power(math.e, -L_t_1)
    C_t_1 = np.sum(e_L_matrix)
    strategy_matrix[:, t] = e_L_matrix/C_t_1




#quesition b
e_Z_matrix = np.power(math.e, -adversary_matrix)
lm = -np.log(np.sum(strategy_matrix*e_Z_matrix, axis=0))  # the mix loss for each time step
# print(lm)
# print(sum(lm))

#question_c
expert_matrix_0 = [[1, 1, 1, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]]
expert_matrix_0 = np.array(expert_matrix_0)

expert_matrix_1 = [[0, 0, 0, 0],
                  [1, 1, 1, 1],
                  [0, 0, 0, 0]]
expert_matrix_1 = np.array(expert_matrix_1)

expert_matrix_2 = [[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 1]]
expert_matrix_2 = np.array(expert_matrix_2)

expert0_loss = np.sum(-np.log(np.sum(expert_matrix_0*e_Z_matrix, axis=0)))
expert1_loss = np.sum(-np.log(np.sum(expert_matrix_1*e_Z_matrix, axis=0)))
expert2_loss = np.sum(-np.log(np.sum(expert_matrix_2*e_Z_matrix, axis=0)))

expert_loss = np.min([expert0_loss, expert1_loss, expert2_loss])


expert_regrets = np.sum(-np.log(np.sum(strategy_matrix*e_Z_matrix, axis=0))) - expert_loss
# print(expert_regrets)
#
# #question_e and f
C = np.log(3)+ expert_loss
