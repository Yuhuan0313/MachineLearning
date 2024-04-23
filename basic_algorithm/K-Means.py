import numpy as np


class Kmeans:
    def __init__(self, data, K, t_max):
        self.data = data
        self.K = K
        self.t_max = t_max

    def dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def run(self):
        data_size = np.size(self.data, 0)
        idxs = np.random.choice(data_size, self.K, replace=False)
        clus_center = [self.data[idx] for idx in idxs]
        for t in range(self.t_max):
            clus = [[] for _ in range(self.K)]
            for i in range(data_size):
                d = [self.dist(np.array(clus_center[j]), np.array((self.data[i]))) for j in range(self.K)]
                idx = np.argmin(d)
                clus[idx].append(self.data[i])
            num = 0
            for j in range(self.K):
                temp = np.mean(clus[j], 0)
                if sum(clus_center[j] - temp) == 0:
                    num += 1
                    clus_center[j] = temp
            if num == self.K:
                break
        return clus_center,clus
