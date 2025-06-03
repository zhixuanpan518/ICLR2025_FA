import numpy as np
import math

vectors = np.random.randn(3072, 10)

Q, _ = np.linalg.qr(vectors)

cluster = Q[:, :10]

for i in range(5):
    cluster[:, i] = cluster[:, i] * (6+2*i) * (30.72)**0.5
for i in range(5, 10):
    cluster[:, i] = cluster[:, i] * (-4+2*i) * (30.72)**0.5

print(np.linalg.norm(cluster, axis=0))

np.save('cluster.npy',cluster)

dataset = np.zeros((1000, 3072))
print(cluster.shape)
for i in range(10):
    data = np.random.randn(100, 3072) + cluster[:,i]
    dataset[i*100:(i+1)*100] = data

np.save('train_data.npy', dataset)

for i in range(10):
    data = np.random.randn(100, 3072) + cluster[:,i]  
    dataset[i*100:(i+1)*100] = data

np.save('test_data.npy', dataset)

