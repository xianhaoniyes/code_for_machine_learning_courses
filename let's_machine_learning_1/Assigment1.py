import matplotlib.pyplot as plt
import numpy as np

# x= np.arange(-10, 10.0, 0.01)
# y = abs(1+x)+ abs(2+x)+abs(2+x)
# plt.plot(x, y)
# plt.xticks(np.arange(-10,10,1))
# plt.yticks(np.arange(5,30,1))
# print(np.min(y))
#
#
#
# plt.show()
# x = [1,2,3]
# print ((x))

# apparent_error = np.load('apparent_error_for_1000.npy')
# true_error = np.load('true_error_for_1000.npy')
# plt.plot(apparent_error,label = 'apparent_error')
# x = [0,10,100,1000,10000]
# xi = [i for i in range (0,5)]
# plt.plot(true_error,label = 'estimated_true_error')
# plt.yticks(np.arange(0 ,0.6,step = 0.05))
# plt.xticks(xi,x)
#
# plt.xlabel('lambda_value')
# plt.ylabel('error')
# plt.title('Regularization Curve for 1000 Samples')
# plt.legend(loc='lower right')
# plt.show()

# medians_0_10 = np.load ('medians_0_1000.npy')
# plt.plot(medians_0_10[0],'ro',label ='m+')
# plt.plot(medians_0_10[1],'bo',label = 'm-')
# plt.legend(loc = 'upper right')
# xi = [i for i in range (0,21)]
# plt.xticks(xi)
# plt.ylabel('values')
# plt.xlabel('dimension')
# plt.title('medians of 1000 samples with lambda 0')
# print(np.std(medians_0_10[0]))
# print(np.std(medians_0_10[1]))
#
# plt.show()


results_0 = np.load('results_0.npy')
results_1 = np.load('results_1.npy')
results_2 = np.load('results_2.npy')

plt.plot(results_0, 'ro', label = 'neural_network')
plt.plot(results_1, 'bo', label = 'TPT')
plt.plot(results_2,'go',label = 'neural_network_based_TPT')
plt.ylabel('auc')
plt.xlabel('number')
plt.legend()
plt.show()

print(np.mean(results_0), np.std(results_0))
print(np.mean(results_1), np.std(results_1))
print(np.mean(results_2), np.std(results_2))
