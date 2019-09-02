import numpy as np
import matplotlib.pyplot as plt



se_acc = np.load('se_acc.npy')
se_square = np.load('se_square.npy')

co_acc = np.load('co_acc.npy')
co_square = np.load('co_square.npy')




x = [0, 8, 16, 32, 64, 128, 256, 512]

plt.xticks(x)
plt.ylim(0.35, 0.5)
plt.plot(x, se_acc,marker = 'o')
plt.xlabel('Size of Unlabeled Samples')
plt.ylabel('Expected True Error')
plt.title('Learning Curve for Self_training ')
plt.show()

plt.clf()
plt.xticks(x)
plt.ylim(0.35, 0.5)
plt.plot(x, co_acc,marker = 'o')
plt.xlabel('Size of Unlabeled Samples')
plt.ylabel('Expected True Error')
plt.title('Learning Curve for Co_training ')
plt.show()


plt.clf()
plt.figure(figsize=(8,5))
plt.xticks(x)

# plt.ylim(0.35, 0.5)
plt.plot(x, se_square,marker= 'o' )
plt.xlabel('Size of Unlabeled Samples')
plt.ylabel('Expected Squared loss')
plt.title('Learning Curve for se_training ')
plt.show()


plt.figure(figsize=(8,5))
plt.clf()
plt.xticks(x)

# plt.ylim(0.35, 0.5)
plt.plot(x, co_square,marker = 'o')
plt.xlabel('Size of Unlabeled Samples')
plt.ylabel('Expected Squared loss')
plt.title('Learning Curve for co_training ')
plt.show()


print(se_square)
print(co_square)



